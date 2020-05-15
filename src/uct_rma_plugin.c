/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
#include <inttypes.h>

#include "core.h"
#include "ibvwrap.h"
#include "nccl.h"
#include "nccl_net.h"
#include "p2p_plugin.h"
#include "param.h"
#include "socket.h"
#include <uct/api/uct.h>

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

NCCL_PARAM(UCTDisable, "UCT_DISABLE", 0);

#define NCCL_UCT_MAX_COMM_RKEYS 16
extern ncclDebugLogger_t pluginLogFunction;

static int ncclNIbDevs = -1;


struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
struct userIbDev userIbDevs[MAX_IB_DEVS];

ncclResult_t nccl_uct_devices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t nccl_uct_get_properties(int dev, ncclNetProperties_t* props)
{
  return nccl_p2p_ib_get_properties(ncclIbDevs, dev, props);
}

pthread_mutex_t nccl_uct_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct nccl_uct_listen_handle {
  union socketAddress connectAddr;
} nccl_uct_listen_handle_t;

typedef struct nccl_uct_listen_comm {
  int           dev;
  int           fd;
} nccl_uct_listen_comm_t;


static union socketAddress nccl_ucx_if_addr;
static char if_name[MAX_IF_NAME_SIZE];

static ncclResult_t get_socket_addr(union socketAddress *addr)
{
  memcpy(addr, &nccl_ucx_if_addr, sizeof(*addr));
  return ncclSuccess;
}


ncclResult_t nccl_uct_init(ncclDebugLogger_t logFunction)
{
  if (ncclParamUCTDisable()) return ncclInternalError;
  return nccl_p2p_ib_init(&ncclNIbDevs, ncclIbDevs, if_name, &nccl_ucx_if_addr,
                          NULL, logFunction);
}

enum {
  NCCL_UCT_REQ_TYPE_SEND,
  NCCL_UCT_REQ_TYPE_RECV,
};

typedef struct nccl_uct_request {
  uct_completion_t compl;
  int              used;
  int              type;
  int              done;
  int              size;
  int              free;
  uct_worker_h     worker;
} nccl_uct_request_t;

typedef struct nccl_uct_ctx {
  int                 fd;
  int                 ready;
  int                 num_mh;
  nccl_uct_request_t  reqs[MAX_REQUESTS];
  ucs_async_context_t *async;
  uct_component_h     component;
  uct_worker_h        worker;
  uct_md_h            md;
  uct_md_attr_t       md_attr;
  uct_iface_h         iface;
  uct_iface_attr_t    iface_attr;
  uct_ep_h            ep;
} nccl_uct_ctx_t;

typedef struct nccl_uct_send_fifo {
  uint64_t addr;
  uint64_t addr_request;
  int      size;
  uint32_t seq;
  uint32_t ready;
  int      rkey_id;
  char     rkey_buf[32];
} nccl_uct_send_fifo_t;

typedef struct nccl_uct_rem_fifo {
  nccl_uct_send_fifo_t elems[MAX_REQUESTS];
  uint64_t             addr;
  uct_rkey_bundle_t    rkey;
  uint32_t             tail;
} nccl_uct_rem_fifo_t;

typedef struct nccl_uct_rkey_cache {
  int               exist;
  uct_rkey_bundle_t rkey;
} nccl_uct_rkey_cache_t;

typedef struct nccl_uct_send_comm {
  nccl_uct_ctx_t        super;
  uint32_t              fifo_head;
  nccl_uct_send_fifo_t  fifo[MAX_REQUESTS];
  uct_mem_h             fifo_memh;
  uct_rkey_bundle_t     rem_req_rkey;
  nccl_uct_rkey_cache_t rkey_cache[NCCL_UCT_MAX_COMM_RKEYS];
} nccl_uct_send_comm_t;

typedef struct nccl_uct_recv_comm {
  nccl_uct_ctx_t      super;
  nccl_uct_rem_fifo_t rem_fifo;
  uct_mem_h           req_memh;
} nccl_uct_recv_comm_t;

static ucs_status_t init_iface(char *dev_name, char *tl_name,
                               nccl_uct_ctx_t *ctx) {
  ucs_status_t       status;
  uct_iface_config_t *config;
  uct_iface_params_t params;

  params.field_mask           = UCT_IFACE_PARAM_FIELD_OPEN_MODE |
                                UCT_IFACE_PARAM_FIELD_DEVICE |
                                UCT_IFACE_PARAM_FIELD_STATS_ROOT |
                                UCT_IFACE_PARAM_FIELD_CPU_MASK;
  params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
  params.mode.device.tl_name  = tl_name;
  params.mode.device.dev_name = dev_name;
  params.stats_root           = NULL;
  UCS_CPU_ZERO(&params.cpu_mask);

  status = uct_md_iface_config_read(ctx->md, tl_name, NULL, NULL, &config);
  if (status != UCS_OK) {
      WARN("Failed to read iface config");
      return status;
  }

  status = uct_iface_open(ctx->md, ctx->worker, &params, config, &ctx->iface);
  uct_config_release(config);
  if (status != UCS_OK) {
      WARN("Failed to open iface");
      return status;
  }
  uct_iface_progress_enable(ctx->iface, UCT_PROGRESS_SEND |
                                         UCT_PROGRESS_RECV);

  status = uct_iface_query(ctx->iface, &ctx->iface_attr);
  if (status != UCS_OK) {
      WARN("Failed to query iface");
      uct_iface_close(ctx->iface);
  }

  return status;
}


static ucs_status_t nccl_uct_tl_init(nccl_uct_ctx_t *ctx, int dev)
{
  ucs_status_t           status;
  unsigned               num_components;
  unsigned               i, j, k;
  uct_component_h        *components;
  uct_component_attr_t   component_attr;
  uct_md_config_t        *md_config;
  uct_tl_resource_desc_t *tl_resources;
  unsigned               num_tl_resources;
  char                   *tl_name;
  char                   dev_name[MAXNAMESIZE];

  tl_name = getenv("NCCL_UCT_TL");
  if (!tl_name) {
      tl_name = "rc_verbs";
  }
  snprintf(dev_name, MAXNAMESIZE, "%s:%u", ncclIbDevs[dev].devName,
           ncclIbDevs[dev].port);

  status = uct_query_components(&components, &num_components);
  if (status != UCS_OK) {
      WARN("Failed to query uct components");
      return ncclInternalError;
  }
  for (i = 0; i < num_components; i++) {
    component_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT;
    status = uct_component_query(components[i], &component_attr);
    if (status != UCS_OK) {
      WARN("Failed to query component attrs");
      goto release_component_list;
    }

    component_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
    component_attr.md_resources = alloca(sizeof(*component_attr.md_resources) *
                                         component_attr.md_resource_count);
    status = uct_component_query(components[i], &component_attr);
    if (status != UCS_OK) {
      WARN("Failed to query md resources");
      goto release_component_list;
    }

    ctx->iface = NULL;
    for (j = 0; j < component_attr.md_resource_count; j++) {
      status = uct_md_config_read(components[i], NULL, NULL, &md_config);
      if (status != UCS_OK) {
        WARN("Failed to read md config");
        goto release_component_list;
      }

      status = uct_md_open(components[i],
                           component_attr.md_resources[j].md_name,
                           md_config, &ctx->md);
      uct_config_release(md_config);
      if (status != UCS_OK) {
        WARN("Failed to open md");
        goto release_component_list;
      }

      status = uct_md_query(ctx->md, &ctx->md_attr);
      if (status != UCS_OK) {
        WARN("Failed to query md");
        goto close_md;
      }

      status = uct_md_query_tl_resources(ctx->md, &tl_resources,
                                         &num_tl_resources);
      if (status != UCS_OK) {
        WARN("Failed to query tl resources");
        goto close_md;
      }

      for (k = 0; k < num_tl_resources; k++) {
        if (!strcmp(dev_name, tl_resources[k].dev_name) &&
            !strcmp(tl_name, tl_resources[k].tl_name)) {
          status = init_iface(dev_name, tl_resources[k].tl_name, ctx);
          if (status != UCS_OK) {
            break;
          }
          
          ctx->component = components[i];
          INFO(NCCL_NET, "Using "UCT_TL_RESOURCE_DESC_FMT,
                         UCT_TL_RESOURCE_DESC_ARG(&tl_resources[k]));
          goto release_tl_resources;
                
          }
      }
release_tl_resources:
      uct_release_tl_resource_list(tl_resources);
      if ((status == UCS_OK) && (k < num_tl_resources)) {
        goto release_component_list;
      }
    }
  }
  WARN("No supported (dev/tl) found (%s/%s)\n",
        dev_name, tl_name);
  status = UCS_ERR_UNSUPPORTED;

release_component_list:
  uct_release_component_list(components);
  return status;
close_md:
  uct_md_close(ctx->md);
  goto release_component_list;
}

typedef struct nccl_uct_ctx_addr {
  size_t            addr_len;
  void              *addr;
  size_t            dev_addr_len;
  uct_device_addr_t *dev_addr;
} nccl_uct_ctx_addr_t;

ncclResult_t nccl_uct_init_ctx(int dev, nccl_uct_ctx_t *ctx,
                               nccl_uct_ctx_addr_t **ctx_addr)
{
  nccl_uct_ctx_addr_t *addr = calloc(1, sizeof(nccl_uct_ctx_addr_t));
  uct_ep_params_t     ep_params;

  UCXCHECK(ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK, &ctx->async));
  UCXCHECK(uct_worker_create(ctx->async, UCS_THREAD_MODE_SINGLE, &ctx->worker));
  UCXCHECK(nccl_uct_tl_init(ctx, dev));

  addr->dev_addr_len = ctx->iface_attr.device_addr_len;
  addr->dev_addr     = calloc(1, addr->dev_addr_len);
  UCXCHECK(uct_iface_get_device_address(ctx->iface, addr->dev_addr));
  if (ctx->iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
    INFO(NCCL_NET, "iface connect to iface");
  }
  else {
    INFO(NCCL_NET, "ep connect to ep");
    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = ctx->iface;
    UCXCHECK(uct_ep_create(&ep_params, &ctx->ep));
    addr->addr_len  = ctx->iface_attr.ep_addr_len;
    addr->addr = calloc(1, addr->addr_len);
    UCXCHECK(uct_ep_get_address(ctx->ep, addr->addr));
  }

  *ctx_addr = addr;

  return ncclSuccess;
}

ncclResult_t nccl_uct_send_ctx_addr(int fd, nccl_uct_ctx_addr_t *ctx_addr)
{
  NCCLCHECK(socketSend(fd, &ctx_addr->addr_len, sizeof(ctx_addr->addr_len)));
  NCCLCHECK(socketSend(fd, ctx_addr->addr, ctx_addr->addr_len));
  NCCLCHECK(socketSend(fd, &ctx_addr->dev_addr_len, sizeof(ctx_addr->dev_addr_len)));
  NCCLCHECK(socketSend(fd, ctx_addr->dev_addr, ctx_addr->dev_addr_len));
  
  return ncclSuccess;
}

ncclResult_t nccl_uct_recv_ctx_addr(int fd, nccl_uct_ctx_addr_t **ctx_addr)
{
  nccl_uct_ctx_addr_t *addr = calloc(1, sizeof(nccl_uct_ctx_addr_t));

  NCCLCHECK(socketReceive(fd, &addr->addr_len, sizeof(addr->addr_len)));
  addr->addr = malloc(addr->addr_len);
  NCCLCHECK(socketReceive(fd, addr->addr, addr->addr_len));
  NCCLCHECK(socketReceive(fd, &addr->dev_addr_len, sizeof(addr->dev_addr_len)));
  addr->dev_addr = malloc(addr->dev_addr_len);
  NCCLCHECK(socketReceive(fd, addr->dev_addr, addr->dev_addr_len));
  
  *ctx_addr = addr;
  return ncclSuccess;
}


void nccl_uct_free_ctx_addr(nccl_uct_ctx_addr_t *ctx_addr)
{
  free(ctx_addr->dev_addr);
  free(ctx_addr->addr);
}

ncclResult_t nccl_uct_listen(int dev, void *handle, void **listen_comm)
{
  nccl_uct_listen_handle_t *my_handle = (nccl_uct_listen_handle_t*)handle;
  nccl_uct_listen_comm_t   *comm;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(nccl_uct_listen_comm_t)));
  memset(comm, 0, sizeof(nccl_uct_listen_comm_t));
  NCCL_STATIC_ASSERT(sizeof(nccl_uct_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE,
                     "UCT-RMA listen handle size too large");
  NCCLCHECK(get_socket_addr(&(my_handle->connectAddr)));
  NCCLCHECK(createListenSocket(&comm->fd, &my_handle->connectAddr));

  comm->dev = dev;
  *listen_comm = comm;

  return ncclSuccess;
}

ncclResult_t nccl_uct_connect(int dev, void *handle, void **send_comm)
{
  nccl_uct_listen_handle_t *recv_handle = (nccl_uct_listen_handle_t*)handle;
  nccl_uct_send_comm_t     *comm;
  nccl_uct_ctx_addr_t      *my_ctx_addr;
  uint64_t                 fifo_adr;
  int                      fifo_len;
  void                     *rkey_buf;
  int                      rkey_buf_len;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(*comm)));
  NCCLCHECK(connectAddress(&comm->super.fd, &recv_handle->connectAddr));

  NCCLCHECK(nccl_uct_init_ctx(dev, &comm->super, &my_ctx_addr));
  NCCLCHECK(nccl_uct_send_ctx_addr(comm->super.fd, my_ctx_addr));
  nccl_uct_free_ctx_addr(my_ctx_addr);

  fifo_adr = (uint64_t)comm->fifo;
  fifo_len = sizeof(nccl_uct_send_fifo_t) * (MAX_REQUESTS);
  UCXCHECK(uct_md_mem_reg(comm->super.md, (void*)fifo_adr, fifo_len,
                          UCT_MD_MEM_ACCESS_RMA,
                          &comm->fifo_memh));
  rkey_buf_len = comm->super.md_attr.rkey_packed_size;
  rkey_buf = alloca(rkey_buf_len);
  UCXCHECK(uct_md_mkey_pack(comm->super.md, comm->fifo_memh, rkey_buf));
  NCCLCHECK(socketSend(comm->super.fd, rkey_buf, rkey_buf_len));
  NCCLCHECK(socketSend(comm->super.fd, &fifo_adr, sizeof(uint64_t)));

  *send_comm = comm;
  return ncclSuccess;
}

ncclResult_t nccl_uct_accept(void *listen_comm, void **recv_comm)
{
  nccl_uct_listen_comm_t *l_comm        = (nccl_uct_listen_comm_t*)listen_comm;
  nccl_uct_recv_comm_t   *comm          = (nccl_uct_recv_comm_t*)recv_comm;
  socklen_t              socklen        = sizeof(struct sockaddr_in);
  nccl_uct_ctx_addr_t    *my_ctx_addr, *peer_ctx_addr;
  struct sockaddr_in     sockaddr;
  void                   *rkey_buf;
  int                    rkey_buf_len;
  uint64_t               req_adr;
  uint64_t               req_len;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(nccl_uct_recv_comm_t)));
  SYSCHECKVAL(accept(l_comm->fd, (struct sockaddr*)&sockaddr, &socklen),
                     "accept", comm->super.fd);

  NCCLCHECK(nccl_uct_init_ctx(l_comm->dev, &comm->super, &my_ctx_addr));
  NCCLCHECK(nccl_uct_recv_ctx_addr(comm->super.fd, &peer_ctx_addr));

  if (comm->super.iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
    UCXCHECK(uct_ep_connect_to_ep(comm->super.ep,
                                  peer_ctx_addr->dev_addr,
                                  peer_ctx_addr->addr));
  }
  NCCLCHECK(nccl_uct_send_ctx_addr(comm->super.fd, my_ctx_addr));
  nccl_uct_free_ctx_addr(my_ctx_addr);
  nccl_uct_free_ctx_addr(peer_ctx_addr);

  rkey_buf_len = comm->super.md_attr.rkey_packed_size;
  rkey_buf = alloca(rkey_buf_len);
  NCCLCHECK(socketReceive(comm->super.fd, rkey_buf, rkey_buf_len));
  NCCLCHECK(socketReceive(comm->super.fd, &comm->rem_fifo.addr, sizeof(uint64_t)));
  UCXCHECK(uct_rkey_unpack(comm->super.component, rkey_buf,
                           &comm->rem_fifo.rkey));
  req_adr = (uint64_t)comm->super.reqs;
  req_len = sizeof(nccl_uct_request_t) * MAX_REQUESTS;
  UCXCHECK(uct_md_mem_reg(comm->super.md, (void*)req_adr, req_len,
                          UCT_MD_MEM_ACCESS_RMA, &comm->req_memh));
  UCXCHECK(uct_md_mkey_pack(comm->super.md, comm->req_memh, rkey_buf));
  NCCLCHECK(socketSend(comm->super.fd, rkey_buf, rkey_buf_len));

  *recv_comm = comm;
  return ncclSuccess;
}

typedef struct nccl_uct_mhandle {
  uct_mem_h uct_mh;
  void      *rkey_buf;
  int       rkey_buf_size;
  int       id;
} nccl_uct_mhandle_t;

#define REG_ALIGN (4096)
ncclResult_t nccl_uct_regmr(void* comm, void* data, int size, int type,
                            void** mhandle)
{
  uint64_t           addr = (uint64_t)data;
  nccl_uct_ctx_t     *ctx = (nccl_uct_ctx_t*)comm;
  nccl_uct_mhandle_t *mh;
  uint64_t           reg_addr, reg_size;

  reg_addr = addr & (~(REG_ALIGN - 1));
  reg_size = addr + size - reg_addr;
  reg_size = ROUNDUP(reg_size, REG_ALIGN);

  //UCT_MD_MEM_FLAG_FIXED || UCT_MD_MEM_FLAG_LOCK
  mh = calloc(1, sizeof(*mh));
  mh->id = ctx->num_mh;
  ctx->num_mh++;
  assert(ctx->num_mh < NCCL_UCT_MAX_COMM_RKEYS);
  mh->rkey_buf = calloc(1, ctx->md_attr.rkey_packed_size);
  UCXCHECK(uct_md_mem_reg(ctx->md, (void*)reg_addr, reg_size,
                          UCT_MD_MEM_ACCESS_RMA, &mh->uct_mh));
  UCXCHECK(uct_md_mkey_pack(ctx->md, mh->uct_mh, mh->rkey_buf));
  mh->rkey_buf_size = ctx->md_attr.rkey_packed_size;
  
  *mhandle = mh;
  return ncclSuccess;
}

ncclResult_t nccl_uct_deregmr(void* comm, void* mhandle)
{
  nccl_uct_ctx_t     *ctx = (nccl_uct_ctx_t*)comm;
  nccl_uct_mhandle_t *mh  = (nccl_uct_mhandle_t*)mhandle;

  UCXCHECK(uct_md_mem_dereg(ctx->md, mh->uct_mh));
  free(mh->rkey_buf);
  free(mh);

  return ncclSuccess;
}

static ncclResult_t nccl_uct_send_check(nccl_uct_send_comm_t *comm)
{
  nccl_uct_ctx_addr_t *peer_ctx_addr;
  void                *rkey_buf;
  int                 rkey_buf_len = comm->super.md_attr.rkey_packed_size;

  NCCLCHECK(nccl_uct_recv_ctx_addr(comm->super.fd, &peer_ctx_addr));

  if (comm->super.iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
    UCXCHECK(uct_ep_connect_to_ep(comm->super.ep,
                                  peer_ctx_addr->dev_addr,
                                  peer_ctx_addr->addr));
  }
  rkey_buf = alloca(rkey_buf_len);
  NCCLCHECK(socketReceive(comm->super.fd, rkey_buf, rkey_buf_len));
  UCXCHECK(uct_rkey_unpack(comm->super.component, rkey_buf,
                           &comm->rem_req_rkey));   
  comm->super.ready = 1;
  NCCLCHECK(socketSend(comm->super.fd, &comm->super.ready, sizeof(int)));
  nccl_uct_free_ctx_addr(peer_ctx_addr);

  return ncclSuccess;
}

static ncclResult_t nccl_uct_recv_check(nccl_uct_recv_comm_t *comm)
{
  int bytes = 0;

  NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, comm->super.fd, &comm->super.ready,
                           sizeof(int), &bytes));
  if (bytes == 0) {
      return ncclSuccess;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, comm->super.fd, &comm->super.ready,
                       sizeof(int), &bytes));

  return ncclSuccess;
}

void nccl_uct_completion_callback(uct_completion_t *self, ucs_status_t status)
{
  nccl_uct_request_t *req = (nccl_uct_request_t*)self;
  req->done = 1;
  return;
}

ncclResult_t nccl_uct_get_request(nccl_uct_request_t* reqs,
                                  nccl_uct_request_t** req)
{
  nccl_uct_request_t *r;
  int                i;

  for (i = 0; i < MAX_REQUESTS; i++) {
    r = reqs + i;
    if (r->used == 0) {
      r->used        = 1;
      r->type        = 0;
      r->done        = 0;
      r->size        = -1;
      r->free        = 0;
      r->compl.count = 1;
      r->compl.func  = nccl_uct_completion_callback;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/UCX_RMA : unable to allocate requests");
  *req = NULL;

  return ncclInternalError;
}

ncclResult_t nccl_uct_isend(void *send_comm, void *data, int size,
                            void *mhandle, void **request)
{
  nccl_uct_send_comm_t          *comm = (nccl_uct_send_comm_t*)send_comm;
  nccl_uct_mhandle_t            *mh   = (nccl_uct_mhandle_t*)mhandle;
  volatile nccl_uct_send_fifo_t *slot;
  nccl_uct_request_t            *req;
  volatile uint32_t             *ready_ptr;
  ucs_status_t                  status;
  uct_iov_t                     iov;
  uct_rkey_bundle_t             rkey;

  if (comm->super.ready == 0) {
    NCCLCHECK(nccl_uct_send_check(comm));
  }
  if (comm->super.ready == 0) {
    *request = NULL;
    return ncclSuccess;
  }

  slot = comm->fifo + (comm->fifo_head % MAX_REQUESTS);
  ready_ptr = &slot->ready;
  if (*ready_ptr == 0) {
    *request = NULL;
    return ncclSuccess;
  }

  NCCLCHECK(nccl_uct_get_request(comm->super.reqs, &req));
  req->size = 0;
  if (comm->rkey_cache[slot->rkey_id].exist == 0) {
    UCXCHECK(uct_rkey_unpack(comm->super.component, (void*)slot->rkey_buf,
                             &comm->rkey_cache[slot->rkey_id].rkey));
    comm->rkey_cache[slot->rkey_id].exist = 1;
  }

  iov.buffer = data;
  iov.length = size;
  iov.memh   = mh->uct_mh;
  iov.stride = 0;
  iov.count  = 1;
  status = uct_ep_put_zcopy(comm->super.ep, &iov, 1, slot->addr,
                            comm->rkey_cache[slot->rkey_id].rkey.rkey,
                            &req->compl);
  if (status != UCS_INPROGRESS) {
      req->done = 1;
  }
  status = uct_ep_put_short(comm->super.ep, &req->size, sizeof(int),
                            slot->addr_request, comm->rem_req_rkey.rkey);
  slot->ready = 0;
  comm->fifo_head++;

  req->worker = comm->super.worker;
  req->type   = NCCL_UCT_REQ_TYPE_SEND;
  *request = req;
  return ncclSuccess;
}


ncclResult_t nccl_uct_post_fifo(nccl_uct_recv_comm_t *comm, nccl_uct_mhandle_t *mh,
                                uint64_t addr, int size, uint64_t req_addr)
{
  nccl_uct_send_fifo_t *local_elem;
  uint64_t             remote_addr;
  ucs_status_t         status;

  local_elem = comm->rem_fifo.elems + (comm->rem_fifo.tail % MAX_REQUESTS);
  local_elem->addr         = addr;
  local_elem->ready        = 1;
  local_elem->size         = size;
  local_elem->seq          = comm->rem_fifo.tail;
  local_elem->addr_request = req_addr;
  local_elem->rkey_id      = mh->id;
  
  memcpy(local_elem->rkey_buf, mh->rkey_buf, mh->rkey_buf_size);
  remote_addr = comm->rem_fifo.addr + (comm->rem_fifo.tail % MAX_REQUESTS) *
                                      sizeof(nccl_uct_send_fifo_t);
  status = uct_ep_put_short(comm->super.ep, local_elem, sizeof(nccl_uct_send_fifo_t),
                            remote_addr, comm->rem_fifo.rkey.rkey);
  if (status < 0) {
      WARN("post_fifo put short failed %d", (int)status);
      return ncclInternalError;
  }
  comm->rem_fifo.tail++;

  return ncclSuccess;
}

ncclResult_t nccl_uct_irecv(void *recv_comm, void *data, int size,
                            void *mhandle, void **request)
{
  nccl_uct_recv_comm_t *comm = (nccl_uct_recv_comm_t*)recv_comm;
  nccl_uct_request_t   *req;

  if (comm->super.ready == 0) {
    NCCLCHECK(nccl_uct_recv_check(comm));
  }
  if (comm->super.ready == 0) {
    *request = NULL;
    return ncclSuccess;
  }

  NCCLCHECK(nccl_uct_get_request(comm->super.reqs, &req));
  req->size = -1;
  NCCLCHECK(nccl_uct_post_fifo(comm, mhandle, (uint64_t)data, size,
                               (uint64_t)&req->size));
  req->worker = comm->super.worker;
  req->type   = NCCL_UCT_REQ_TYPE_RECV;
  *request = req;
  return ncclSuccess;
}
ncclResult_t nccl_uct_flush(void* recv_comm, void* data, int size,
                            void* mhandle)
{
  return ncclSuccess;
}

ncclResult_t nccl_uct_test(void *request, int *done, int *size)
{
  nccl_uct_request_t *req = (nccl_uct_request_t*)request;

  uct_worker_progress(req->worker);
  switch(req->type) {
    case NCCL_UCT_REQ_TYPE_SEND:
      if (req->done == 1) {
        req->used = 0;
        *done = 1;
        return ncclSuccess;
      }
      break;
    case NCCL_UCT_REQ_TYPE_RECV:
      if (req->size != -1) {
        req->used = 0;
        *done = 1;
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


ncclResult_t nccl_uct_close_send(void *send_comm)
{
  nccl_uct_send_comm_t *comm = (nccl_uct_send_comm_t*)send_comm;
  int                  i;
  if (comm) {
    for(i = 0; i < NCCL_UCT_MAX_COMM_RKEYS; i++) {
      if (comm->rkey_cache[i].exist == 1) {
        uct_rkey_release(comm->super.component,
                         &comm->rkey_cache[i].rkey);
      }
    }
    uct_md_mem_dereg(comm->super.md, comm->fifo_memh);
    uct_ep_destroy(comm->super.ep);
    uct_iface_close(comm->super.iface);
    uct_md_close(comm->super.md);
    uct_worker_destroy(comm->super.worker);
    ucs_async_context_destroy(comm->super.async);
  }  

  return ncclSuccess;
}

ncclResult_t nccl_uct_close_recv(void *recv_comm)
{
  nccl_uct_recv_comm_t *comm = (nccl_uct_recv_comm_t*)recv_comm;

  if (comm) {
    uct_md_mem_dereg(comm->super.md, comm->req_memh);
    uct_rkey_release(comm->super.component, &comm->rem_fifo.rkey);
    uct_ep_destroy(comm->super.ep);
    uct_iface_close(comm->super.iface);
    uct_md_close(comm->super.md);
    uct_worker_destroy(comm->super.worker);
    ucs_async_context_destroy(comm->super.async);
  }  

  return ncclSuccess;
}

ncclResult_t nccl_uct_close_listen(void *listen_comm)
{
  nccl_uct_listen_comm_t *comm = (nccl_uct_listen_comm_t *)listen_comm;

  if (comm) {
    close(comm->fd);
    free(comm);
  }

  return ncclSuccess;
}

ncclNet_t uctRmaPlugin = {
  "UCT",
  nccl_uct_init,
  nccl_uct_devices,
  nccl_uct_get_properties,
  nccl_uct_listen,
  nccl_uct_connect,
  nccl_uct_accept,
  nccl_uct_regmr,
  nccl_uct_deregmr,
  nccl_uct_isend,
  nccl_uct_irecv,
  nccl_uct_flush,
  nccl_uct_test,
  nccl_uct_close_send,
  nccl_uct_close_recv,
  nccl_uct_close_listen
};
