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

/*
 * If request == REQUEST_COMPLETED_ZERO_LENGTGH:
 *  ucp_send or ucp_recv was completed immediately and worker progress is not needed
 *  message size == 0 and gpu flush is not needed
 * 
 * If request == REQUEST_COMPLETED_NON_ZERO_LENGTH:
 *  ucp_send or ucp_recv was completed immediately and worker progres is not needed
 *  message size > 0 and gpu flush is needed
 * 
 * If request != REQUEST_COMPLETED_ZERO_LENGTGH and request != REQUEST_COMPLETED_NON_ZERO_LENGTH:
 *  normal ucp request.
 */
enum {
  REQUEST_COMPLETED_ZERO_LENGTGH    = 1,
  REQUEST_COMPLETED_NON_ZERO_LENGTH = 2
};

typedef struct ucx_rma_mhandle {
  ucp_mem_h  ucp_memh;
  ucp_rkey_h rkey;
  void       *rkey_buf;
  size_t     rkey_buf_size;
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

/**
 * Listen handle that is sent from receiver to sender through OOB connection
 */
typedef struct ucx_listen_handle {
  union socketAddress connectAddr; /* reciever socket address */
} ucx_listen_handle_t;

/**
 * Listen commincator for UCX plugin.
 */
typedef struct ucx_listen_comm {
  int           dev;    /* device number in ncclIbDevs which will
                         * be used to recieve data */
  int           fd;     /* Socket fd */
  ucp_context_h ctx;    /* ucp_context associated with specific device dev */
  ucp_worker_h  worker; /* ucx_worker created on ctx, worker can be shared between
                           multiple connections */
} ucx_rma_listen_comm_t;

typedef struct connect_msg {
  size_t addr_len;
} connect_msg_t;

struct ep_list {
  int            fd;
  struct ep_list *next;
};

/**
 * Connection descriptor. Used to store all opened connections.
 */
struct nccl_ucx_worker {
  ucp_context_h  ctx;      /* ucp_context bounded to specific device */
  ucp_worker_h   worker;   /* ucp worker associated with ctx */
  int            count;    /* number of connections that uses this worker */
  struct ep_list *eps;     /* oob conection to all endpoints that were opened on this worker */
  ucp_tag_t      last_tag; /* tag that last created connection uses */
};
static struct nccl_ucx_worker workers[MAX_IB_DEVS];

typedef struct ucx_gpu_flush {
  int      enabled;
  int      hostMem;
  ucp_ep_h flush_ep;
} ucx_gpu_flush_t;

/**
 * Common data member for ucx_send_comm and ucx_recv_comm.
 * Used to map/unmap memory in nccl_ucx_regmr/nccl_ucx_deregmr
 */
typedef struct ucx_ctx {
  ucp_context_h     ucp_ctx;
  ucx_gpu_flush_t   gpuFlush;
  ucx_rma_mhandle_t *mh[8]; 
  int               num_mh;
} ucx_ctx_t;

enum {
  UCX_RMA_REQ_TYPE_SEND,
  UCX_RMA_REQ_TYPE_RECV
};

typedef struct ucx_rma_request {
  int              used;
  int              type;
  int              done;
  int              size;
  int              free;
  ucs_status_ptr_t st;
  ucp_worker_h     worker;
} ucx_rma_request_t;

typedef struct ucx_rma_send_fifo {
  uint64_t addr;
  uint64_t addr_request;
  int      size;
  uint32_t seq;
  uint32_t ready;
  int      rkey_idx;
  char     rkey_buf[40];
} ucx_rma_send_fifo_t;

/**
 * Sender communicator
 */
typedef struct ucx_rma_send_comm {
  ucp_context_h       ctx;        /* ucp_context bounded to specific device */
  ucx_gpu_flush_t     gpuFlush;   /* flushing handle */
  ucx_rma_mhandle_t   *mh[8]; 
  int                 num_mh;
  ucp_worker_h        worker;     /* ucp worker associated with ctx */
  ucp_ep_h            ep;         /* ucp endpoint created on worker */
  int                 fd;         /* socket fd for OOB connection */
  int                 ready;      /* indicates that send communicator is fully initialized */
  ucx_rma_send_fifo_t fifo[MAX_REQUESTS];
  ucx_rma_request_t   reqs[MAX_REQUESTS];
  uint32_t            fifo_head;
  ucp_mem_h           fifo_memh;
  ucp_rkey_h          rkey;
  uint64_t            rem_req_addr;
  ucp_rkey_h          rem_key[8];
} ucx_rma_send_comm_t;

typedef struct ucx_rma_rem_fifo {
  ucx_rma_send_fifo_t elems[MAX_REQUESTS];
  uint64_t            addr;
  ucp_rkey_h          rkey;
  uint32_t            tail;
} ucx_rma_rem_fifo_t;


typedef struct ucx_rma_recv_comm {
  ucp_context_h      ctx;
  ucx_gpu_flush_t    gpuFlush;
  ucx_rma_mhandle_t  *mh[8]; 
  int                num_mh;
  ucp_worker_h       worker;
  ucp_ep_h           ep;
  int                fd;
  int                ready;
  ucx_rma_request_t  reqs[MAX_REQUESTS];
  ucx_rma_rem_fifo_t rem_fifo;
  ucp_mem_h          req_memh;
} ucx_rma_recv_comm_t;


static union socketAddress nccl_ucx_if_addr;
static char if_name[MAX_IF_NAME_SIZE];

static ncclResult_t get_socket_addr(union socketAddress *addr) {
  memcpy(addr, &nccl_ucx_if_addr, sizeof(*addr));
  return ncclSuccess;
}

static ncclResult_t ucx_init_context(ucp_context_h *ctx, int dev) {
  ucp_params_t ucp_params;
  ucp_config_t *config;
  char         ucx_dev_name[PATH_MAX];

  snprintf(ucx_dev_name, PATH_MAX, "%s:%d", ncclIbDevs[dev].devName, ncclIbDevs[dev].port);
  UCXCHECK(ucp_config_read("NCCL", NULL, &config));
  UCXCHECK(ucp_config_modify(config, "NET_DEVICES", ucx_dev_name));
  UCXCHECK(ucp_config_modify(config, "TLS", "ib"));
  UCXCHECK(ucp_config_modify(config, "ZCOPY_THRESH", "4"));

  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES;
  ucp_params.features     = UCP_FEATURE_RMA;

  UCXCHECK(ucp_init(&ucp_params, config, ctx));
  ucp_config_release(config);

  return ncclSuccess;
}

static ncclResult_t ucx_init_worker(ucp_context_h ctx, ucp_worker_h *worker) {
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

static ncclResult_t ucx_worker_get_netaddress(ucp_worker_h worker, ucp_address_t **address, size_t *address_length) {
  ucp_worker_attr_t attr;

  attr.field_mask    = UCP_WORKER_ATTR_FIELD_ADDRESS |
                       UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
  attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;

  UCXCHECK(ucp_worker_query(worker, &attr));
  *address = malloc(attr.address_length);
  if (address == NULL) {
    return ncclSystemError;
  }

  memcpy(*address, attr.address, attr.address_length);
  *address_length = attr.address_length;
  free(attr.address);

  return ncclSuccess;
}

#define UCX_SHARED_WORKER
static ncclResult_t ucx_get_ctx_and_worker(int dev, ucp_context_h *ctx, ucp_worker_h *worker) {
  pthread_mutex_lock(&nccl_ucx_rma_lock);
#ifdef UCX_SHARED_WORKER
  if (ncclNIbDevs < dev) {
    WARN("Device index is too large");
    return ncclSystemError;
  }

  if (workers[dev].count == 0) {
    ucx_init_context(&workers[dev].ctx, dev);
    ucx_init_worker(workers[dev].ctx, &workers[dev].worker);
  }

  *ctx    = workers[dev].ctx;
  *worker = workers[dev].worker;

  ucp_worker_progress(*worker);
  workers[dev].count++;
#else
  ucx_init_context(ctx, dev);
  ucx_init_worker(*ctx, worker);
#endif
  pthread_mutex_unlock(&nccl_ucx_rma_lock);
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_free_worker(ucp_worker_h worker) {
  int i;
  int dummy;
  struct ep_list *ep, *cur;

  pthread_mutex_lock(&nccl_ucx_rma_lock);
  for(i = 0; i < ncclNIbDevs; i++) {
    if (worker == workers[i].worker) {
      workers[i].count--;
      if (workers[i].count == 0){
        ep = workers[i].eps;
        while(ep){
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

static ncclResult_t nccl_ucx_add_ep(ucp_worker_h worker, int fd) {
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

  return nccl_p2p_ib_init(&ncclNIbDevs, ncclIbDevs, if_name, &nccl_ucx_if_addr, NULL, logFunction);
}

ncclResult_t nccl_ucx_rma_listen(int dev, void *handle, void **listen_comm) {
  ucx_listen_handle_t   *my_handle = (ucx_listen_handle_t*)handle;
  ucx_rma_listen_comm_t *comm;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(ucx_rma_listen_comm_t)));
  memset(comm, 0, sizeof(ucx_rma_listen_comm_t));

  NCCL_STATIC_ASSERT(sizeof(ucx_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE,
                     "UCX-RMA listen handle size too large");
  NCCLCHECK(get_socket_addr(&(my_handle->connectAddr)));
  NCCLCHECK(createListenSocket(&comm->fd, &my_handle->connectAddr));
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker));

  comm->dev = dev; 
  *listen_comm = comm;
 
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_connect(int dev, void *handle, void **send_comm)
{
  ucx_listen_handle_t  *recv_handle = (ucx_listen_handle_t*)handle;
  ucx_rma_send_comm_t  *comm;
  ucp_address_t        *my_addr;
  size_t               local_addr_len;
  ucp_mem_map_params_t mmap_params;
  size_t               rkey_buf_size;
  void                 *rkey_buf;
  uint64_t             fifo_adr;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(*comm)));
  memset(comm, 0, sizeof(*comm));

  NCCLCHECK(connectAddress(&comm->fd, &recv_handle->connectAddr));
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker));
  NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));

  NCCLCHECK(socketSend(comm->fd, &local_addr_len, sizeof(size_t)));
  NCCLCHECK(socketSend(comm->fd, my_addr, local_addr_len));
  
  free(my_addr);
  fifo_adr = (uint64_t)comm->fifo;
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  mmap_params.address    = (void*)fifo_adr;
  mmap_params.length     = sizeof(ucx_rma_send_fifo_t) *
                           MAX_REQUESTS;
  ucp_mem_map(comm->ctx, &mmap_params, &comm->fifo_memh);
  ucp_rkey_pack(comm->ctx, comm->fifo_memh, &rkey_buf, &rkey_buf_size);
  NCCLCHECK(socketSend(comm->fd, &rkey_buf_size, sizeof(size_t)));
  NCCLCHECK(socketSend(comm->fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketSend(comm->fd, &fifo_adr, sizeof(uint64_t)));
  ucp_rkey_buffer_release(rkey_buf);
  *send_comm = comm;
  comm->num_mh = 0;

//   comm->gpuFlush.enabled = 0;
//   nccl_ucx_add_ep(comm->worker,comm->fd);
//   INFO(NCCL_NET, "Worker address length: %zu", local_addr_len);

//   NCCLCHECK(ncclIbMalloc((void**)&comm->fifo_tail, sizeof(uint32_t)));


  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_accept(void *listen_comm, void **recv_comm)
{
  ucx_rma_listen_comm_t *l_comm = (ucx_rma_listen_comm_t *)listen_comm;
  socklen_t             socklen = sizeof(struct sockaddr_in);
  ucx_rma_recv_comm_t   *r_comm;
  ucp_address_t         *peer_addr, *my_addr;
  ucp_ep_params_t       ep_params;
  struct sockaddr_in    sockaddr;
  size_t                peer_addr_len, local_addr_len;
  void                  *rkey_buf;
  size_t                rkey_buf_size;

  NCCLCHECK(ncclIbMalloc((void**)&r_comm, sizeof(ucx_rma_recv_comm_t)));
  memset(r_comm, 0, sizeof(ucx_rma_recv_comm_t));
  r_comm->ctx    = l_comm->ctx;
  r_comm->worker = l_comm->worker;

  SYSCHECKVAL(accept(l_comm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", r_comm->fd);
  NCCLCHECK(socketReceive(r_comm->fd, &peer_addr_len, sizeof(size_t)));
  peer_addr = malloc(peer_addr_len);
  if (peer_addr == NULL) {
    return ncclSystemError;
  }
  NCCLCHECK(socketReceive(r_comm->fd, peer_addr, peer_addr_len));

  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address    = peer_addr;
  UCXCHECK(ucp_ep_create(r_comm->worker, &ep_params, &r_comm->ep));
  NCCLCHECK(ucx_worker_get_netaddress(r_comm->worker, &my_addr, &local_addr_len));
  NCCLCHECK(socketSend(r_comm->fd, &local_addr_len, sizeof(size_t)));
  NCCLCHECK(socketSend(r_comm->fd, my_addr, local_addr_len));

  free(my_addr);

  NCCLCHECK(socketReceive(r_comm->fd, &rkey_buf_size, sizeof(size_t)));

  rkey_buf = malloc(rkey_buf_size);
  if (rkey_buf == NULL) {
    return ncclSystemError;
  }
  NCCLCHECK(socketReceive(r_comm->fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketReceive(r_comm->fd, &r_comm->rem_fifo.addr, sizeof(uint64_t)));

  UCXCHECK(ucp_ep_rkey_unpack(r_comm->ep, rkey_buf, &r_comm->rem_fifo.rkey));
  free(rkey_buf);


  uint64_t             req_adr;
  ucp_mem_map_params_t mmap_params;


  req_adr = (uint64_t)r_comm->reqs;
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  mmap_params.address    = (void*)req_adr;
  mmap_params.length     = sizeof(ucx_rma_request_t) *
                           MAX_REQUESTS;
  ucp_mem_map(r_comm->ctx, &mmap_params, &r_comm->req_memh);
  ucp_rkey_pack(r_comm->ctx, r_comm->req_memh, &rkey_buf, &rkey_buf_size);
  NCCLCHECK(socketSend(r_comm->fd, &rkey_buf_size, sizeof(size_t)));
  NCCLCHECK(socketSend(r_comm->fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketSend(r_comm->fd, &req_adr, sizeof(uint64_t)));
  ucp_rkey_buffer_release(rkey_buf);



  r_comm->gpuFlush.enabled = (nccl_p2p_gdr_support(l_comm->dev) == ncclSuccess);  
  if (r_comm->gpuFlush.enabled) {
    ucp_address_t *my_addr;
    size_t        local_addr_len;

    NCCLCHECK(ucx_worker_get_netaddress(r_comm->worker, &my_addr, &local_addr_len));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = my_addr;
    UCXCHECK(ucp_ep_create(r_comm->worker, &ep_params, &r_comm->gpuFlush.flush_ep));
    free(my_addr);
  }

  free(peer_addr);
  r_comm->num_mh = 0;
  *recv_comm = r_comm;

  return ncclSuccess;
}

#define REG_ALIGN (4096)
ncclResult_t nccl_ucx_rma_regmr(void* comm, void* data, int size, int type, void** mhandle)
{
  ucx_ctx_t            *ctx = (ucx_ctx_t*)comm;
  uint64_t             addr = (uint64_t)data;
  ucp_mem_map_params_t mmap_params;
  ucx_rma_mhandle_t    *mh;
  uint64_t             reg_addr, reg_size;
  size_t               rkey_buf_size;
  void                 *rkey_buf;
  
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

  UCXCHECK(ucp_mem_map(ctx->ucp_ctx, &mmap_params, &mh->ucp_memh));
  UCXCHECK(ucp_rkey_pack(ctx->ucp_ctx, mh->ucp_memh, &mh->rkey_buf,
                         &mh->rkey_buf_size));

  if (ctx->gpuFlush.enabled) {
    UCXCHECK(ucp_ep_rkey_unpack(ctx->gpuFlush.flush_ep, mh->rkey_buf, &mh->rkey));
  }
  
  ctx->mh[ctx->num_mh] = mh;
  *mhandle = (void*)ctx->num_mh;
  ctx->num_mh++;
//  *mhandle = mh;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_deregmr(void* comm, void* mhandle) {
  ucx_ctx_t         *ctx = (ucx_ctx_t*)comm;
  ucx_rma_mhandle_t *mh  = ctx->mh[(int)mhandle];

  if (ctx->gpuFlush.enabled) {
      ucp_rkey_destroy(mh->rkey);
  }


  ucp_mem_unmap(ctx->ucp_ctx, mh->ucp_memh);
  free(mh);

  return ncclSuccess;
}


ncclResult_t ucx_rma_get_request(ucx_rma_request_t* reqs,
                                 ucx_rma_request_t** req)
{
  ucx_rma_request_t *r;
  int               i;

  for (i=0; i < MAX_REQUESTS; i++) {
    r = reqs + i;
    if (r->used == 0) {
      r->used = 1;
      r->type = 0;
      r->done = 0;
      r->size = -1;
      r->free = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/UCX_RMA : unable to allocate requests");
  *req = NULL;

  return ncclInternalError;
}

static ncclResult_t nccl_ucx_rma_send_check(ucx_rma_send_comm_t *comm)
{
  int             bytes;
  size_t          peer_addr_len;
  ucp_address_t   *peer_addr;
  ucp_ep_params_t ep_params;
  void            *rkey_buf;
  size_t          rkey_buf_size;


  bytes = 0;
  NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, comm->fd, &peer_addr_len,
                           sizeof(size_t), &bytes));
  if (bytes == 0) {
      return ncclSuccess;
  }

  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, comm->fd, &peer_addr_len,
                       sizeof(size_t), &bytes));
  
  peer_addr = malloc(peer_addr_len);
  if (peer_addr == NULL) {
    return ncclSystemError;
  }
  NCCLCHECK(socketReceive(comm->fd, peer_addr, peer_addr_len));

  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address    = peer_addr;
  UCXCHECK(ucp_ep_create(comm->worker, &ep_params, &comm->ep));
  
  NCCLCHECK(socketReceive(comm->fd, &rkey_buf_size, sizeof(size_t)));

  rkey_buf = malloc(rkey_buf_size);
  if (rkey_buf == NULL) {
    return ncclSystemError;
  }
  NCCLCHECK(socketReceive(comm->fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketReceive(comm->fd, &comm->rem_req_addr, sizeof(uint64_t)));

  UCXCHECK(ucp_ep_rkey_unpack(comm->ep, rkey_buf, &comm->rkey));
  

  comm->ready = 1;
  NCCLCHECK(socketSend(comm->fd, &comm->ready, sizeof(int)));
  free(peer_addr);
  free(rkey_buf);
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_recv_check(ucx_rma_recv_comm_t *comm)
{
  int bytes = 0;
  NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, comm->fd, &comm->ready,
                           sizeof(int), &bytes));
  if (bytes == 0) return ncclSuccess;
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, comm->fd, &comm->ready,
                       sizeof(int), &bytes));

  return ncclSuccess;
}

static void send_cb(void *request, ucs_status_t status) {
  return;
}


ncclResult_t nccl_ucx_rma_isend(void *send_comm, void *data, int size,
                                void *mhandle, void **request)
{
  ucx_rma_send_comm_t          *comm = (ucx_rma_send_comm_t*)send_comm;
  volatile ucx_rma_send_fifo_t *slot;
  volatile uint32_t            *ready_ptr;
  ucx_rma_request_t            *req;
  ucs_status_ptr_t             st;
  ucp_rkey_h                   rkey;

  if (comm->ready == 0) {
    NCCLCHECK(nccl_ucx_rma_send_check(comm));
  }
  if (comm->ready == 0) {
    *request = NULL;
    return ncclSuccess;
  }

  slot = comm->fifo + (comm->fifo_head % MAX_REQUESTS);
  ready_ptr = &slot->ready;
  if (*ready_ptr == 0) {
    ucp_worker_progress(comm->worker);
    *request = NULL;
    return ncclSuccess;
  }

  NCCLCHECK(ucx_rma_get_request(comm->reqs, &req));
  req->size = size;
//  INFO(NCCL_NET, "send addr %p compl %p", (void*)slot->addr, (void*)slot->addr_request);;
  if (comm->rem_key[slot->rkey_idx] == NULL) {
    UCXCHECK(ucp_ep_rkey_unpack(comm->ep, (void*)slot->rkey_buf, &comm->rem_key[slot->rkey_idx]));
  }

  ucp_put_nbi(comm->ep, data, size, slot->addr, comm->rem_key[slot->rkey_idx]);
  req->st = ucp_put_nb(comm->ep, &req->size, sizeof(int),
                       slot->addr_request, comm->rkey, send_cb);
  ucp_worker_progress(comm->worker);

//  ucp_rkey_destroy(comm->rem_key[slot->rkey_idx]);
  slot->ready = 0;
  slot->addr  = 0ULL;
  slot->size  = 0;
  slot->seq   = 0;
  comm->fifo_head++;

  req->worker = comm->worker;
  req->type   = UCX_RMA_REQ_TYPE_SEND;
  *request = req;
  return ncclSuccess;
}

ncclResult_t ucx_rma_post_fifo(ucx_rma_recv_comm_t *comm, ucx_rma_mhandle_t *mh,
                               uint64_t addr, int size, void *req_addr)
{
  ucx_rma_send_fifo_t *local_elem;
  uint64_t            remote_addr;

  local_elem = comm->rem_fifo.elems + (comm->rem_fifo.tail % MAX_REQUESTS);
  local_elem->addr         = addr;
  local_elem->ready        = 1;
  local_elem->size         = size;
  local_elem->seq          = comm->rem_fifo.tail;
  local_elem->addr_request = (uint64_t)req_addr;
  local_elem->rkey_idx     = (int)mh;

  memcpy(local_elem->rkey_buf, comm->mh[(int)mh]->rkey_buf, comm->mh[(int)mh]->rkey_buf_size);
  remote_addr = comm->rem_fifo.addr + (comm->rem_fifo.tail % MAX_REQUESTS) *
                                      sizeof(ucx_rma_send_fifo_t);
  
 // INFO(NCCL_NET, "postf addr %p compl %p rem_fifo %p", (void*)addr, req_addr, remote_addr);
  ucp_put_nbi(comm->ep, (void*)local_elem, sizeof(ucx_rma_send_fifo_t),
              remote_addr, comm->rem_fifo.rkey);
  comm->rem_fifo.tail++;

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_irecv(void *recv_comm, void *data, int size,
                                void *mhandle, void **request) {
  ucx_rma_recv_comm_t *comm = (ucx_rma_recv_comm_t*)recv_comm;
  ucx_rma_mhandle_t   *mh   = (ucx_rma_mhandle_t*)mhandle;
  ucx_rma_request_t   *req;

  if (comm->ready == 0) {
    NCCLCHECK(nccl_ucx_rma_recv_check(comm));
  }

  if (comm->ready == 0) {
    *request = NULL;
    return ncclSuccess;
  }
  
  NCCLCHECK(ucx_rma_get_request(comm->reqs, &req));
  req->size = size;

  NCCLCHECK(ucx_rma_post_fifo(comm, mh, (uint64_t)data, size, (void*)&req->size));
  ucp_worker_progress(comm->worker);
  req->size   = -1;
  req->worker = comm->worker;
  req->type   = UCX_RMA_REQ_TYPE_RECV;

  *request = req;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_flush(void* recv_comm, void* data, int size, void* mhandle) {
  ucx_rma_recv_comm_t *comm = (ucx_rma_recv_comm_t *)recv_comm;
  ucx_rma_request_t   *req;

  if ((comm->gpuFlush.enabled == 0) || (size == 0)) {
    return ncclSuccess;
  }

  req = ucp_get_nb(comm->gpuFlush.flush_ep, &comm->gpuFlush.hostMem, 1,
                   (uint64_t)data, comm->mh[(int)mhandle]->rkey, send_cb);
  if (UCS_PTR_IS_ERR(req)) {
    WARN("ucx_flush: unable to read data (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  } else if (req != NULL) {
    while(ucp_request_check_status(req) == UCS_INPROGRESS) {
       ucp_worker_progress(comm->worker);
    }
    ucp_request_free(req);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_test(void *request, int *done, int *size)
{
  ucx_rma_request_t *req = (ucx_rma_request_t*)request;

  ucp_worker_progress(req->worker);
  switch(req->type){
    case UCX_RMA_REQ_TYPE_SEND:
      if ((req->st == NULL) || (ucp_request_check_status(req->st) != UCS_INPROGRESS)) {
        *done = 1;
        if (req->st) {
          ucp_request_free(req->st);
        }
        req->used = 0;
        return ncclSuccess;

      }
    break;
    case UCX_RMA_REQ_TYPE_RECV:
      if (req->size != -1) {
        *done = 1;
        req->used = 0;
        if (size) {
          *size = req->size;
        }

        return ncclSuccess;
      }
    break;
  };

  *done = 0;
  return ncclSuccess;
}

static void wait_close(ucp_worker_h worker, ucx_rma_request_t *req)
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
  ucx_rma_send_comm_t *comm      = (ucx_rma_send_comm_t*) send_comm;
  void                *close_req;
  int                 i;

  if (send_comm){
    ucp_mem_unmap(comm->ctx, comm->fifo_memh);
    if (comm->ready) {
      ucp_rkey_destroy(comm->rkey);
    }

    for (i = 0; i < comm->num_mh; i++) {
      if (comm->rem_key[i]) {
        ucp_rkey_destroy(comm->rem_key[i]);
      }
    }
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->worker, close_req);
      int close = 1;
      NCCLCHECK(socketSend(comm->fd, &close, sizeof(int)));
    }
    nccl_ucx_free_worker(comm->worker);
    free(comm);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_close_recv(void *recv_comm)
{
  ucx_rma_recv_comm_t *comm      = (ucx_rma_recv_comm_t*)recv_comm;
  void                *close_req;

  if (recv_comm){
    ucp_mem_unmap(comm->ctx, comm->req_memh);
    ucp_rkey_destroy(comm->rem_fifo.rkey);
    if (comm->gpuFlush.enabled) {
      close_req = ucp_ep_close_nb(comm->gpuFlush.flush_ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->worker, close_req);
    }
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->worker, close_req);
      int close=1;
      NCCLCHECK(socketSend(comm->fd, &close, sizeof(int)));  
    }
    nccl_ucx_free_worker(comm->worker);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_close_listen(void *listen_comm)
{
  ucx_rma_listen_comm_t *comm = (ucx_rma_listen_comm_t *)listen_comm;

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
