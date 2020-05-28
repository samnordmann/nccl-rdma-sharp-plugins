/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <pthread.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>

#include "core.h"
#include "ibvwrap.h"
#include "nccl.h"
#include "nccl_net.h"
#include "p2p_plugin.h"
#include "param.h"
#include "socket.h"
#include "ucp/api/ucp.h"


#define UCXCHECK(cmd) do {                           \
  int e = cmd;                                       \
  if( UCS_OK != e ) {                                \
    WARN("Failed: UCX error %s:%d '%d' %s\n",        \
        __FILE__,__LINE__, e, ucs_status_string(e)); \
    return ncclInternalError;                        \
  }                                                  \
} while(0)

#define UCXCHECK_VOID(cmd) do {                      \
  int e = cmd;                                       \
  if( UCS_OK != e ) {                                \
    WARN("Failed: UCX error %s:%d '%d' %s\n",        \
        __FILE__,__LINE__, e, ucs_status_string(e)); \
  }                                                  \
} while(0)

NCCL_PARAM(UCXRMADisable, "UCX_RMA_DISABLE", 0);

extern ncclDebugLogger_t pluginLogFunction;

static int ncclNIbDevs = -1;

typedef struct ucx_rma_mhandle {
  ucp_mem_h  ucp_memh;
  ucp_rkey_h rkey;
  void       *rkey_buf;
  size_t     rkey_buf_size;
  int        index;
} ucx_rma_mhandle_t;

struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
struct userIbDev userIbDevs[MAX_IB_DEVS];

ncclResult_t nccl_ucx_rma_devices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_get_properties(int dev, ncclNetProperties_t* props)
{
  return nccl_p2p_ib_get_properties(ncclIbDevs, dev, props);
}

pthread_mutex_t nccl_ucx_rma_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct ucx_rma_listen_handle {
  union socketAddress connectAddr;
} ucx_rma_listen_handle_t;

typedef struct nccl_ucx_rma_listen_comm {
  int dev;
  int fd;
} nccl_ucx_rma_listen_comm_t;

struct ep_list {
  int            fd;
  struct ep_list *next;
};

struct nccl_ucx_worker {
  ucp_context_h  ctx;
  ucp_worker_h   worker;
  int            count;
  struct ep_list *eps;
};
static struct nccl_ucx_worker workers[MAX_IB_DEVS];

typedef struct ucx_gpu_flush {
  int      enabled;
  int      hostMem;
  ucp_ep_h flush_ep;
} ucx_gpu_flush_t;

enum {
  UCX_RMA_REQ_TYPE_SEND,
  UCX_RMA_REQ_TYPE_RECV
};

typedef struct nccl_ucx_rma_request {
  char             ucx_req[256];
  int              used;
  int              type;
  int              done;
  int              size;
  int              free;
  uint64_t         am_msg;
  int              seq;
  ucs_status_ptr_t st;
  ucp_worker_h     worker;
} nccl_ucx_rma_request_t;

typedef struct ucx_rma_send_fifo {
  uint64_t addr;
  uint64_t addr_request;
  int      size;
  uint32_t seq;
  uint32_t ready;
  int      rkey_idx;
  int      req_id;
  char     rkey_buf[40];
} ucx_rma_send_fifo_t;

typedef struct nccl_ucx_rma_ctx {
  int                    id;
  int                    fd;
  int                    ready;
  ucs_status_ptr_t       ep_ready;
  ucp_context_h          ctx;
  ucp_worker_h           worker;
  ucx_gpu_flush_t        gpuFlush;
  uint64_t               num_mh;
  ucx_rma_mhandle_t      *mh[16];
  nccl_ucx_rma_request_t reqs[MAX_REQUESTS];
} nccl_ucx_rma_ctx_t;

typedef struct nccl_ucx_rma_send_comm {
  nccl_ucx_rma_ctx_t  super;
  ucp_ep_h            ep;
  ucx_rma_send_fifo_t fifo[MAX_REQUESTS];
  uint32_t            fifo_head;
  ucp_mem_h           fifo_memh;
  ucp_rkey_h          rem_key[16];
  int                 rem_am_id;
} nccl_ucx_rma_send_comm_t;

typedef struct ucx_rma_rem_fifo {
  ucx_rma_send_fifo_t elems[MAX_REQUESTS];
  uint64_t            addr;
  ucp_rkey_h          rkey;
  uint32_t            tail;
} ucx_rma_rem_fifo_t;

typedef struct nccl_ucx_rma_recv_comm {
  nccl_ucx_rma_ctx_t     super;
  ucp_ep_h               ep;
  ucx_rma_rem_fifo_t     rem_fifo;
} nccl_ucx_rma_recv_comm_t;


static union socketAddress nccl_ucx_if_addr;
static char if_name[MAX_IF_NAME_SIZE];

static ncclResult_t get_socket_addr(union socketAddress *addr)
{
  memcpy(addr, &nccl_ucx_if_addr, sizeof(*addr));
  return ncclSuccess;
}

typedef struct nccl_ucx_am_request {
  nccl_ucx_rma_request_t *req;
} nccl_ucx_am_request_t;

static ncclResult_t nccl_ucx_rma_init_ucp(int dev, ucp_context_h *ctx)
{
  ucp_params_t ucp_params;
  ucp_config_t *config;
  char         ucx_dev_name[PATH_MAX];

  snprintf(ucx_dev_name, PATH_MAX, "%s:%d", ncclIbDevs[dev].devName,
           ncclIbDevs[dev].port);
  UCXCHECK(ucp_config_read("NCCL", NULL, &config));
  UCXCHECK(ucp_config_modify(config, "NET_DEVICES", ucx_dev_name));
  UCXCHECK(ucp_config_modify(config, "TLS", "ib"));
  UCXCHECK(ucp_config_modify(config, "ZCOPY_THRESH", "128"));

  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                            UCP_PARAM_FIELD_REQUEST_SIZE;
  ucp_params.features     = UCP_FEATURE_RMA |
                            UCP_FEATURE_AM;
  ucp_params.request_size = sizeof(nccl_ucx_am_request_t);

  UCXCHECK(ucp_init(&ucp_params, config, ctx));
  ucp_config_release(config);

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_init_worker(ucp_context_h ctx,
                                             ucp_worker_h *worker)
{
  ucp_worker_params_t worker_params;
  ucp_worker_attr_t   worker_attr;

  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

  UCXCHECK(ucp_worker_create(ctx, &worker_params, worker));

  worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
  ucp_worker_query(*worker, &worker_attr);
  if (worker_attr.thread_mode != UCS_THREAD_MODE_MULTI) {
    INFO(NCCL_NET, "Thread mode multi is not supported");
  }

  return ncclSuccess;
}

#define UCX_RMA_USE_SHARED_WORKER
static ncclResult_t nccl_ucx_rma_init_comm_context(int dev,
                                                   nccl_ucx_rma_ctx_t *comm_ctx)
{
  pthread_mutex_lock(&nccl_ucx_rma_lock);
#ifdef UCX_RMA_USE_SHARED_WORKER
  if (workers[dev].count == 0) {
    nccl_ucx_rma_init_ucp(dev, &workers[dev].ctx);
    nccl_ucx_rma_init_worker(workers[dev].ctx, &workers[dev].worker);
    workers->count = 0;
    workers->eps   = NULL;
  }

  comm_ctx->ctx    = workers[dev].ctx;
  comm_ctx->worker = workers[dev].worker;
  comm_ctx->id     = workers[dev].count;
  workers[dev].count++;
#else
  nccl_ucx_rma_init_ucp(dev, &comm_ctx->ctx);
  nccl_ucx_rma_init_worker(comm_ctx->ctx, &comm_ctx->worker);
#endif
  pthread_mutex_unlock(&nccl_ucx_rma_lock);
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_send_worker_address(ucp_worker_h worker, int fd)
{
  ucp_worker_attr_t attr;

  attr.field_mask    = UCP_WORKER_ATTR_FIELD_ADDRESS |
                       UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
  attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;

  UCXCHECK(ucp_worker_query(worker, &attr));
  NCCLCHECK(socketSend(fd, &attr.address_length, sizeof(attr.address_length)));
  NCCLCHECK(socketSend(fd, attr.address, attr.address_length));

  free(attr.address);
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_free_worker(ucp_worker_h worker)
{
  int i;
  int dummy;
  struct ep_list *ep, *cur;

  pthread_mutex_lock(&nccl_ucx_rma_lock);
  for(i = 0; i < ncclNIbDevs; i++) {
    if (worker == workers[i].worker) {
      workers[i].count--;
      if (workers[i].count == 0) {
        ep = workers[i].eps;
        while(ep) {
          cur = ep;
          NCCLCHECK(socketReceive(ep->fd, &dummy, sizeof(int)));
          ep = ep->next;
          close(cur->fd);
          free(cur);
        }
        ucp_worker_destroy(workers[i].worker);
        ucp_cleanup(workers[i].ctx);
        workers[i].eps    = NULL;
        workers[i].worker = NULL;
        workers[i].ctx    = NULL;
      }
      break;
    }
  }
  pthread_mutex_unlock(&nccl_ucx_rma_lock);

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_add_ep(ucp_worker_h worker, int fd)
{
  ncclResult_t status = ncclSuccess;
  int i;

  for(i = 0; i < ncclNIbDevs; i++) {
    if (worker == workers[i].worker) {
      struct ep_list *new_ep = (struct ep_list*)malloc(sizeof(struct ep_list));

      if (new_ep == NULL) {
        status = ncclSystemError;
        break;
      }

      new_ep->fd   = fd;
      new_ep->next = workers[i].eps;
      workers[i].eps = new_ep;
      break;
    }
  }

  return status;
}

ncclResult_t nccl_ucx_rma_init(ncclDebugLogger_t logFunction)
{
  if (ncclParamUCXRMADisable()) return ncclInternalError;

  return nccl_p2p_ib_init(&ncclNIbDevs, ncclIbDevs, if_name, &nccl_ucx_if_addr,
                          NULL, logFunction);
}

ncclResult_t nccl_ucx_rma_listen(int dev, void *handle, void **listen_comm)
{
  ucx_rma_listen_handle_t *my_handle = (ucx_rma_listen_handle_t*)handle;
  nccl_ucx_rma_listen_comm_t   *comm;

  NCCL_STATIC_ASSERT(sizeof(ucx_rma_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE,
                     "UCX-RMA listen handle size too large");

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(nccl_ucx_rma_listen_comm_t)));
  NCCLCHECK(get_socket_addr(&(my_handle->connectAddr)));
  NCCLCHECK(createListenSocket(&comm->fd, &my_handle->connectAddr));

  comm->dev = dev; 
  *listen_comm = comm;
 
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_connect(int dev, void *handle, void **send_comm)
{
  ucx_rma_listen_handle_t *recv_handle = (ucx_rma_listen_handle_t*)handle;
  nccl_ucx_rma_send_comm_t      *comm;
  ucp_mem_map_params_t    mmap_params;
  size_t                  rkey_buf_size;
  void                    *rkey_buf;
  uint64_t                fifo_adr;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(*comm)));
  NCCLCHECK(connectAddress(&comm->super.fd, &recv_handle->connectAddr));
  NCCLCHECK(nccl_ucx_rma_init_comm_context(dev, &comm->super));
  NCCLCHECK(nccl_ucx_rma_send_worker_address(comm->super.worker, comm->super.fd));

  fifo_adr = (uint64_t)comm->fifo;
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  mmap_params.address    = (void*)fifo_adr;
  mmap_params.length     = sizeof(ucx_rma_send_fifo_t) *
                           MAX_REQUESTS;
  ucp_mem_map(comm->super.ctx, &mmap_params, &comm->fifo_memh);
  ucp_rkey_pack(comm->super.ctx, comm->fifo_memh, &rkey_buf, &rkey_buf_size);
  NCCLCHECK(socketSend(comm->super.fd, &rkey_buf_size, sizeof(size_t)));
  NCCLCHECK(socketSend(comm->super.fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketSend(comm->super.fd, &fifo_adr, sizeof(uint64_t)));
  ucp_rkey_buffer_release(rkey_buf);
  *send_comm = comm;

  return ncclSuccess;
}

static ucs_status_t nccl_ucx_rma_am_cb(void *arg, void *data, size_t length,
                                       ucp_ep_h reply_ep, unsigned flags)
{
  nccl_ucx_rma_request_t *reqs = (nccl_ucx_rma_request_t*)arg;
  uint64_t *header = data;
  int      size    = *header & 0xFFFFFFFFFFFFFFFF;
  int      id      = *header >>32 ;

  reqs[id].size = size;
  reqs[id].done = 2;

  return UCS_OK;
}

static ncclResult_t nccl_ucx_rma_init_ep(int fd, ucp_worker_h worker, ucp_ep_h *ep)
{
  ucp_ep_params_t ep_params;
  size_t          peer_addr_len;
  void            *peer_addr;

  NCCLCHECK(socketReceive(fd, &peer_addr_len, sizeof(size_t)));
  peer_addr = alloca(peer_addr_len);
  NCCLCHECK(socketReceive(fd, peer_addr, peer_addr_len));

  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address    = peer_addr;
  UCXCHECK(ucp_ep_create(worker, &ep_params, ep));

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_accept(void *listen_comm, void **recv_comm)
{
  nccl_ucx_rma_listen_comm_t *l_comm = (nccl_ucx_rma_listen_comm_t *)listen_comm;
  socklen_t                  socklen = sizeof(struct sockaddr_in);
  nccl_ucx_rma_recv_comm_t   *comm;
  struct sockaddr_in         sockaddr;
  void                       *rkey_buf;
  size_t                     rkey_buf_size;
 
  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(nccl_ucx_rma_recv_comm_t)));
  SYSCHECKVAL(accept(l_comm->fd, (struct sockaddr*)&sockaddr, &socklen),
              "accept", comm->super.fd);
  NCCLCHECK(nccl_ucx_rma_init_comm_context(l_comm->dev, &comm->super));
  UCXCHECK(ucp_worker_set_am_handler(comm->super.worker, comm->super.id,
                                     nccl_ucx_rma_am_cb, comm->super.reqs,0));

  NCCLCHECK(nccl_ucx_rma_init_ep(comm->super.fd, comm->super.worker, &comm->ep));
  NCCLCHECK(nccl_ucx_rma_send_worker_address(comm->super.worker, comm->super.fd));
  NCCLCHECK(socketSend(comm->super.fd, &comm->super.id, sizeof(int)));
  NCCLCHECK(socketReceive(comm->super.fd, &rkey_buf_size, sizeof(size_t)));

  rkey_buf = malloc(rkey_buf_size);
  if (rkey_buf == NULL) {
    return ncclSystemError;
  }
  NCCLCHECK(socketReceive(comm->super.fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketReceive(comm->super.fd, &comm->rem_fifo.addr, sizeof(uint64_t)));
  UCXCHECK(ucp_ep_rkey_unpack(comm->ep, rkey_buf, &comm->rem_fifo.rkey));
  free(rkey_buf);

  if (nccl_p2p_gdr_support(l_comm->dev) == ncclSuccess) {
    comm->super.gpuFlush.enabled = 1;
  }

  if (comm->super.gpuFlush.enabled) {
    ucp_worker_attr_t attr;
    ucp_ep_params_t   ep_params;

    attr.field_mask    = UCP_WORKER_ATTR_FIELD_ADDRESS |
                         UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
    attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;

    UCXCHECK(ucp_worker_query(comm->super.worker, &attr));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = attr.address;
    UCXCHECK(ucp_ep_create(comm->super.worker, &ep_params,
                           &comm->super.gpuFlush.flush_ep));

    free(attr.address);
  }
  comm->super.num_mh = 0;
  *recv_comm = comm;

  return ncclSuccess;
}

#define REG_ALIGN (4096)
ncclResult_t nccl_ucx_rma_regmr(void* comm, void* data, int size, int type,
                                void** mhandle)
{
  nccl_ucx_rma_ctx_t   *ctx = (nccl_ucx_rma_ctx_t*)comm;
  uint64_t             addr = (uint64_t)data;
  ucp_mem_map_params_t mmap_params;
  ucx_rma_mhandle_t    *mh;
  uint64_t             reg_addr, reg_size;
  
  reg_addr = addr & (~(REG_ALIGN - 1));
  reg_size = addr + size - reg_addr;
  reg_size = ROUNDUP(reg_size, REG_ALIGN);

  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH; 
  mmap_params.address    = (void*)reg_addr;
  mmap_params.length     = reg_size;
  
  mh = (ucx_rma_mhandle_t*)malloc(sizeof(ucx_rma_mhandle_t));
  if (mh == NULL) {
    return ncclSystemError;
  }

  UCXCHECK(ucp_mem_map(ctx->ctx, &mmap_params, &mh->ucp_memh));
  UCXCHECK(ucp_rkey_pack(ctx->ctx, mh->ucp_memh, &mh->rkey_buf,
                         &mh->rkey_buf_size));

  if (ctx->gpuFlush.enabled) {
    UCXCHECK(ucp_ep_rkey_unpack(ctx->gpuFlush.flush_ep, mh->rkey_buf, &mh->rkey));
  }
  
  mh->index = ctx->num_mh;
  ctx->mh[ctx->num_mh] = mh;
  ctx->num_mh++;
  *mhandle = mh;

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_deregmr(void* comm, void* mhandle)
{
  nccl_ucx_rma_ctx_t *ctx = (nccl_ucx_rma_ctx_t*)comm;
  ucx_rma_mhandle_t  *mh  = (ucx_rma_mhandle_t*)mhandle;

  if (ctx->gpuFlush.enabled) {
      ucp_rkey_destroy(mh->rkey);
  }
  ucp_mem_unmap(ctx->ctx, mh->ucp_memh);
  free(mh);

  return ncclSuccess;
}

ncclResult_t ucx_rma_get_request(nccl_ucx_rma_request_t* reqs, int* req_id)
{
  nccl_ucx_rma_request_t *r;
  int i;

  for (i = 0; i < MAX_REQUESTS; i++) {
    r = reqs + i;
    if (r->used == 0) {
      r->used = 1;
      r->type = 0;
      r->done = 0;
      r->size = -1;
      r->free = 0;
      r->st   = NULL;
      *req_id = i;
      return ncclSuccess;
    }
  }
  WARN("NET/UCX_RMA: unable to allocate requests");
  *req_id = -1;

  return ncclInternalError;
}

static void nccl_ucx_rma_flush_cb(void *request, ucs_status_t status)
{
  return;
}

enum {
  NCCL_UCX_RMA_COMM_NOT_READY = 0,
  NCCL_UCX_RMA_EP_CREATED     = 1,
  NCCL_UCX_RMA_EP_FLUSHED     = 2,
  NCCL_UCX_RMA_EP_READY       = 3 
};

static ncclResult_t nccl_ucx_rma_send_check(nccl_ucx_rma_send_comm_t *comm)
{
  ucs_status_t st;

  if (comm->super.ready == NCCL_UCX_RMA_COMM_NOT_READY) {
    NCCLCHECK(nccl_ucx_rma_init_ep(comm->super.fd, comm->super.worker, &comm->ep));
    NCCLCHECK(nccl_ucx_add_ep(comm->super.worker, comm->super.fd));
    NCCLCHECK(socketReceive(comm->super.fd, &comm->rem_am_id, sizeof(int)));
    comm->super.ready = NCCL_UCX_RMA_EP_CREATED;
  }

  if (comm->super.ready == NCCL_UCX_RMA_EP_CREATED) {
    comm->super.ep_ready = ucp_ep_flush_nb(comm->ep, 0, nccl_ucx_rma_flush_cb);

    if (comm->super.ep_ready == NULL) {
      comm->super.ready = NCCL_UCX_RMA_EP_READY; 
    } else if (UCS_PTR_IS_ERR(comm->super.ep_ready)) {
      return ncclSystemError;
    } else {
      comm->super.ready = NCCL_UCX_RMA_EP_FLUSHED;
    }
  }

  if (comm->super.ready == NCCL_UCX_RMA_EP_FLUSHED) {
    ucp_worker_progress(comm->super.worker);
    st = ucp_request_check_status(comm->super.ep_ready);
    if (st != UCS_INPROGRESS) {
      ucp_request_free(comm->super.ep_ready);
      comm->super.ready = NCCL_UCX_RMA_EP_READY; 
    }
  }

  if (comm->super.ready == NCCL_UCX_RMA_EP_READY) {
      NCCLCHECK(socketSend(comm->super.fd, &comm->super.ready, sizeof(int)));
  }

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_recv_check(nccl_ucx_rma_recv_comm_t *comm)
{
  int bytes = 0;

  ucp_worker_progress(comm->super.worker);
  NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, comm->super.fd, &comm->super.ready,
                           sizeof(int), &bytes));
  if (bytes == 0) {
    return ncclSuccess;
  }

  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, comm->super.fd, &comm->super.ready,
                       sizeof(int), &bytes));
  NCCLCHECK(nccl_ucx_add_ep(comm->super.worker, comm->super.fd));

  return ncclSuccess;
}

static void nccl_ucx_rma_dummy_cb(void *request, ucs_status_t status)
{
  nccl_ucx_am_request_t *req = (nccl_ucx_am_request_t*)request;

  req->req->done += 1;
  return;
}

static void nccl_ucx_rma_send_cb(void *request, ucs_status_t status, void *data)
{
  nccl_ucx_rma_request_t *req = (nccl_ucx_rma_request_t*)data;

  req->done += 1;
  return;
}

ncclResult_t nccl_ucx_rma_isend(void *send_comm, void *data, int size,
                                void *mhandle, void **request)
{
  nccl_ucx_rma_send_comm_t     *comm = (nccl_ucx_rma_send_comm_t*)send_comm;
  volatile ucx_rma_send_fifo_t *slot;
  volatile uint32_t            *ready_ptr;
  nccl_ucx_rma_request_t       *req;
  ucs_status_ptr_t             st;
  int                          req_id;
  ucp_request_param_t          req_param;

  if (comm->super.ready != NCCL_UCX_RMA_EP_READY) {
    NCCLCHECK(nccl_ucx_rma_send_check(comm));
  }
  if (comm->super.ready != NCCL_UCX_RMA_EP_READY) {
    *request = NULL;
    return ncclSuccess;
  }

  slot = comm->fifo + (comm->fifo_head % MAX_REQUESTS);
  ready_ptr = &slot->ready;
  if (*ready_ptr == 0) {
    ucp_worker_progress(comm->super.worker);
    *request = NULL;
    return ncclSuccess;
  }

  NCCLCHECK(ucx_rma_get_request(comm->super.reqs, &req_id));
  req = &(comm->super.reqs[req_id]);
  req->size = size;
//  if (comm->rem_key[slot->rkey_idx] == NULL) {
    UCXCHECK(ucp_ep_rkey_unpack(comm->ep, (void*)slot->rkey_buf,
                                &comm->rem_key[slot->rkey_idx]));
//  }

  req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                           UCP_OP_ATTR_FIELD_REQUEST  |
                           UCP_OP_ATTR_FIELD_USER_DATA;
  req_param.cb.send      = nccl_ucx_rma_send_cb;
  req_param.user_data    = req;
  req_param.request      = &req->used;
  
  st  = ucp_put_nbx(comm->ep, data, size, slot->addr,
                    comm->rem_key[slot->rkey_idx], &req_param);

  req->done = 0;
  if (UCS_PTR_IS_ERR(st)) {
    WARN("NET/UCX_RMA: isend pub_nb failed");
    return ncclInternalError;
  } else if (st  == NULL) {
    req->done += 1;
  }

  ucp_worker_fence(comm->super.worker);
  req->am_msg = (((uint64_t)slot->req_id) << 32) | ((uint64_t)size);
  req->st = ucp_am_send_nb(comm->ep, comm->rem_am_id, &req->am_msg, 8,
                           ucp_dt_make_contig(1), nccl_ucx_rma_dummy_cb, 0);

  if (req->st == NULL) {
    req->done += 1;
  } else if (UCS_PTR_IS_PTR(req->st)) {
    nccl_ucx_am_request_t *am_req = (nccl_ucx_am_request_t*)req->st;
    am_req->req = req;
  } else {
    WARN("NET/UCX_RMA: isend pub_nb failed");
  }

  ucp_rkey_destroy(comm->rem_key[slot->rkey_idx]);
  comm->rem_key[slot->rkey_idx] = NULL;

  req->seq = slot->seq;
  slot->ready = 0;
  slot->addr  = 0ULL;
  slot->size  = 0;
  slot->seq   = 0;
  comm->fifo_head++;

  req->worker = comm->super.worker;
  req->type   = UCX_RMA_REQ_TYPE_SEND;
  *request = req;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_post_fifo(nccl_ucx_rma_recv_comm_t *comm,
                                    ucx_rma_mhandle_t *mh,
                                    uint64_t addr, int size, int req_id)
{
  ucx_rma_send_fifo_t *local_elem;
  uint64_t            remote_addr;
  ucs_status_t        st;

  local_elem = comm->rem_fifo.elems + (comm->rem_fifo.tail % MAX_REQUESTS);
  local_elem->addr     = addr;
  local_elem->ready    = 1;
  local_elem->size     = size;
  local_elem->seq      = comm->rem_fifo.tail;
  local_elem->rkey_idx = mh->index;
  local_elem->req_id   = req_id;

  memcpy(local_elem->rkey_buf, mh->rkey_buf, mh->rkey_buf_size);
  remote_addr = comm->rem_fifo.addr + (comm->rem_fifo.tail % MAX_REQUESTS) *
                                      sizeof(ucx_rma_send_fifo_t);
  st = ucp_put_nbi(comm->ep, (void*)local_elem, sizeof(ucx_rma_send_fifo_t),
                   remote_addr, comm->rem_fifo.rkey);
  if (st < 0) {
    WARN("ucx_rma post_fifo pub_nbi failed %d", (int)st);
    return ncclInternalError;
  }

  comm->rem_fifo.tail++;

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_irecv(void *recv_comm, void *data, int size,
                                void *mhandle, void **request)
{
  nccl_ucx_rma_recv_comm_t *comm = (nccl_ucx_rma_recv_comm_t*)recv_comm;
  ucx_rma_mhandle_t        *mh   = (ucx_rma_mhandle_t*)mhandle;
  nccl_ucx_rma_request_t   *req;
  int                      req_id;

  if (comm->super.ready != NCCL_UCX_RMA_EP_READY) {
    NCCLCHECK(nccl_ucx_rma_recv_check(comm));
  }

  if (comm->super.ready != NCCL_UCX_RMA_EP_READY) {
    *request = NULL;
    return ncclSuccess;
  }
  
  NCCLCHECK(ucx_rma_get_request(comm->super.reqs, &req_id));
  req = &comm->super.reqs[req_id];

  req->seq = comm->rem_fifo.tail;
  NCCLCHECK(nccl_ucx_rma_post_fifo(comm, mh, (uint64_t)data, size,  req_id));
  ucp_worker_progress(comm->super.worker);
  req->worker = comm->super.worker;
  req->type   = UCX_RMA_REQ_TYPE_RECV;
  *request = req;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_flush(void* recv_comm, void* data, int size,
                                void* mhandle)
{
  nccl_ucx_rma_recv_comm_t *comm = (nccl_ucx_rma_recv_comm_t*)recv_comm;
  ucx_rma_mhandle_t        *mh   = (ucx_rma_mhandle_t*)mhandle;
  nccl_ucx_rma_request_t   *req;

  if ((comm->super.gpuFlush.enabled == 0) || (size == 0)) {
    return ncclSuccess;
  }

  req = ucp_get_nb(comm->super.gpuFlush.flush_ep, &comm->super.gpuFlush.hostMem, 1,
                   (uint64_t)data, mh->rkey, nccl_ucx_rma_flush_cb);
  if (UCS_PTR_IS_ERR(req)) {
    WARN("ucx_flush: unable to read data (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  } else if (req != NULL) {
    while(ucp_request_check_status(req) == UCS_INPROGRESS) {
       ucp_worker_progress(comm->super.worker);
    }
    ucp_request_free(req);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_test(void *request, int *done, int *size)
{
  nccl_ucx_rma_request_t *req = (nccl_ucx_rma_request_t*)request;
  unsigned p;

  *done = 0;
  do {
    if (req->done == 2) {
      *done = 1;
      if (size) {
        *size = req->size;
      }
      if (req->st != NULL) {
        ucp_request_free(req->st);
      }
      req->used = 0;
      return ncclSuccess;
    }

    p = ucp_worker_progress(req->worker);
  } while (p);

  return ncclSuccess;
}

static void wait_close(ucp_worker_h worker, nccl_ucx_rma_request_t *req)
{
  ucs_status_t status;

  if (UCS_PTR_IS_PTR(req)) {
    do {
      ucp_worker_progress(worker);
      status = ucp_request_check_status(req);
    } while(status == UCS_INPROGRESS);
    ucp_request_free(req);
  } else if (req != NULL) {
      WARN("Failed to close UCX endpoint");
  }
}

ncclResult_t nccl_ucx_rma_close_send(void *send_comm)
{
  nccl_ucx_rma_send_comm_t *comm = (nccl_ucx_rma_send_comm_t*) send_comm;
  void *close_req;
  int  i;

  if (send_comm) {
    ucp_mem_unmap(comm->super.ctx, comm->fifo_memh);

    for (i = 0; i < comm->super.num_mh; i++) {
      if (comm->rem_key[i]) {
        ucp_rkey_destroy(comm->rem_key[i]);
      }
    }
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->super.worker, close_req);
      int close = 1;
      NCCLCHECK(socketSend(comm->super.fd, &close, sizeof(int)));
    }
    nccl_ucx_free_worker(comm->super.worker);
    free(comm);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_close_recv(void *recv_comm)
{
  nccl_ucx_rma_recv_comm_t *comm = (nccl_ucx_rma_recv_comm_t*)recv_comm;
  void *close_req;

  if (recv_comm) {
    ucp_rkey_destroy(comm->rem_fifo.rkey);
    if (comm->super.gpuFlush.enabled) {
      close_req = ucp_ep_close_nb(comm->super.gpuFlush.flush_ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->super.worker, close_req);
    }
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->super.worker, close_req);
      int close=1;
      NCCLCHECK(socketSend(comm->super.fd, &close, sizeof(int)));  
    }
    nccl_ucx_free_worker(comm->super.worker);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_close_listen(void *listen_comm)
{
  nccl_ucx_rma_listen_comm_t *comm = (nccl_ucx_rma_listen_comm_t *)listen_comm;

  if (comm) {
    close(comm->fd);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclNet_t ucxRmaPlugin = {
  "UCX_RMA",
  nccl_ucx_rma_init,
  nccl_ucx_rma_devices,
  nccl_ucx_rma_get_properties,
  nccl_ucx_rma_listen,
  nccl_ucx_rma_connect,
  nccl_ucx_rma_accept,
  nccl_ucx_rma_regmr,
  nccl_ucx_rma_deregmr,
  nccl_ucx_rma_isend,
  nccl_ucx_rma_irecv,
  nccl_ucx_rma_flush,
  nccl_ucx_rma_test,
  nccl_ucx_rma_close_send,
  nccl_ucx_rma_close_recv,
  nccl_ucx_rma_close_listen
};
