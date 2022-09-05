/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <sys/time.h>
#include <sys/types.h>

#include "config.h"
#include "core.h"
#include "nccl.h"
#include "nccl_net.h"
#include "p2p_plugin.h"
#include "param.h"
#include "ucc/api/ucc.h"
#include "ucc/api/ucc_version.h"
#include "utils.h"

#define ncclUCC_CHECK(call, msg)                                               \
  do {                                                                         \
    ucc_status_t st = call;                                                    \
    if (st != UCC_OK) {                                                        \
      WARN("UCC: " msg "\n");                                                  \
      return ncclInternalError;                                                \
    }                                                                          \
  } while (0);

extern ncclNet_t NCCL_PLUGIN_SYMBOL;
NCCL_PARAM(UCCGroupSizeThresh, "UCC_GROUP_SIZE_THRESH", 2);

enum ncclUCCRequestType {
  NCCL_UCC_REQ_UCC_COLL,
  NCCL_UCC_REQ_IFLUSH,
};

struct ncclUCCRequest {
  int requestType;
  int size;
  int used;
  ucc_coll_req_h ucc_req;
  ucc_context_h ctx;
};

struct ncclUCCListenComm {
  int dev;
  void *listenCommP2P;
};

struct ncclUCCCollComm {
  int rank;
  int nranks;
  void *recvComm;
  void *sendComm;
  struct ncclUCCRequest *reqs;
  ucc_lib_h lib;
  ucc_context_h ctx;
  ucc_team_h team;
};

struct ncclUCCMemHandle {
  int type;
};

struct ncclUCCInfo {
  uint64_t hostId;
  uint64_t jobId;
};

static __inline__ ucc_datatype_t typeConvert(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
    return UCC_DT_INT8;
  case ncclUint8:
    return UCC_DT_UINT8;
  case ncclInt32:
    return UCC_DT_INT32;
  case ncclUint32:
    return UCC_DT_UINT32;
  case ncclInt64:
    return UCC_DT_INT64;
  case ncclUint64:
    return UCC_DT_UINT64;
  case ncclFloat16:
    return UCC_DT_FLOAT16;
  case ncclFloat32:
    return UCC_DT_FLOAT32;
  case ncclFloat64:
    return UCC_DT_FLOAT64;
  default:
    return -1;
  }
}

static __inline__ ucc_reduction_op_t opConvert(ncclRedOp_t op) {
  switch (op) {
  case ncclSum:
    return UCC_OP_SUM;
  case ncclProd:
    return UCC_OP_PROD;
  case ncclMax:
    return UCC_OP_MAX;
  case ncclMin:
    return UCC_OP_MIN;
  case ncclAvg:
    return UCC_OP_AVG;
  default:
    return -1;
  }
}

int ncclUCCAllGather(void *context, void *src_buf, void *recv_buf, int len) {
  struct ncclUCCCollComm *cComm = (struct ncclUCCCollComm *)context;
  nccl_p2p_plugin_t p2p_plugin;
  void *rMhandle = NULL, *sMhandle = NULL;

  assert(cComm->recvComm != NULL);
  assert(cComm->sendComm != NULL);

  p2p_plugin = nccl_p2p_get_plugin_type();
  if (p2p_plugin != NCCL_P2P_UCX) {
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.regMr(cComm->recvComm, recv_buf,
                                       cComm->nranks * len, NCCL_PTR_HOST,
                                       &rMhandle));
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.regMr(cComm->sendComm, recv_buf,
                                       cComm->nranks * len, NCCL_PTR_HOST,
                                       &sMhandle));
  }

  int speer = cComm->rank;
  memcpy(recv_buf + speer * len, src_buf, len);
  for (int i = 0; i < cComm->nranks - 1; i++) {
    void *srequest = NULL, *rrequest = NULL;
    int rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
      void *rbuf = ((char *)recv_buf) + rpeer * len;
      int tag = 0x69;
      if (srequest == NULL)
        NCCLCHECK(NCCL_PLUGIN_SYMBOL.isend(cComm->sendComm,
                                           ((char *)recv_buf) + speer * len,
                                           len, tag, sMhandle, &srequest));
      if (rrequest == NULL)
        NCCLCHECK(NCCL_PLUGIN_SYMBOL.irecv(cComm->recvComm, 1, &rbuf, &len,
                                           &tag, &rMhandle, &rrequest));
    }
    while (srequest || rrequest) {
      int done;
      if (rrequest)
        NCCLCHECK(NCCL_PLUGIN_SYMBOL.test(rrequest, &done, NULL));
      if (done)
        rrequest = NULL;
      if (srequest)
        NCCLCHECK(NCCL_PLUGIN_SYMBOL.test(srequest, &done, NULL));
      if (done)
        srequest = NULL;
    }
    speer = rpeer;
  }
  if (p2p_plugin != NCCL_P2P_UCX) {
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.deregMr(cComm->recvComm, rMhandle));
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.deregMr(cComm->sendComm, sMhandle));
  }

  return 0;
}

ucc_status_t UCC_oob_allgather(void *src_buf, void *recv_buf, size_t size,
                               void *coll_info, void **request) {
  NCCLCHECK(ncclUCCAllGather(coll_info, src_buf, recv_buf, (int)size))
  return UCC_OK;
}

ucc_status_t UCC_oob_req_test(void *request) { return UCC_OK; }
ucc_status_t UCC_oob_req_free(void *request) { return UCC_OK; }

ncclResult_t ncclUCCInit(ncclDebugLogger_t logFunction) {
  struct timeval tval;
  gettimeofday(&tval, NULL);
  srand((int)tval.tv_usec);

  return NCCL_PLUGIN_SYMBOL.init(logFunction);
}

ncclResult_t ncclUCCDevices(int *ndev) {
  *ndev = ncclNSharpDevs;
  return ncclSuccess;
}

ncclResult_t ncclUCCGetProperties(int dev, ncclNetProperties_t *props) {
  return NCCL_PLUGIN_SYMBOL.getProperties(dev, props);
}

ncclResult_t ncclUCCListen(int dev, void *opaqueHandle, void **listenComm) {
  struct ncclUCCListenComm *lComm;
  ncclResult_t status;

  NCCLCHECK(ncclIbMalloc((void **)&lComm, sizeof(struct ncclUCCListenComm)));
  status = NCCL_PLUGIN_SYMBOL.listen(dev, opaqueHandle, &lComm->listenCommP2P);
  lComm->dev = dev;
  *listenComm = lComm;
  return status;
}

ncclResult_t ncclUCCConnect(void *handles[], int nranks, int rank,
                            void *listenComm, void **collComm) {

  struct ncclUCCListenComm *lComm = (struct ncclUCCListenComm *)listenComm;
  struct ncclUCCCollComm *cComm;
  char *useUCC;
  ucc_status_t status;

  if (nranks < ncclParamUCCGroupSizeThresh()) {
    INFO(NCCL_INIT | NCCL_NET | NCCL_ENV,
         "UCC: Group size:%d is less than threshold:%d. fallback to non-UCC",
         nranks, ncclParamUCCGroupSizeThresh());
    return ncclInvalidUsage;
  }

  useUCC = getenv("NCCL_UCC_DISABLE");
  if (useUCC != NULL) {
    if (strcmp(useUCC, "1") == 0) {
      INFO(NCCL_INIT | NCCL_NET | NCCL_ENV,
           "UCC: Set to disable on this communicator");
      return ncclInvalidUsage;
    }
  }

  NCCLCHECK(ncclIbMalloc((void **)&cComm, sizeof(struct ncclUCCCollComm)));
  NCCLCHECK(ncclIbMalloc((void **)&cComm->reqs,
                         sizeof(struct ncclUCCRequest) * MAX_REQUESTS));

  cComm->nranks = nranks;
  cComm->rank = rank;
  if (cComm->rank == -1) {
    WARN("Could not determine my rank\n");
    return ncclInternalError;
  }
  int next = (cComm->rank + 1) % nranks;
  do {
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.connect(lComm->dev, handles[next],
                                         &cComm->sendComm));
  } while (cComm->sendComm == NULL);

  do {
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.accept(lComm->listenCommP2P,
                                        &cComm->recvComm)); // From prev
  } while (cComm->recvComm == NULL);

  ucc_lib_params_t lib_params = {.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
                                 .thread_mode = UCC_THREAD_MULTIPLE};
  ucc_lib_config_h lib_config;
  ucc_lib_h lib_p;
  ncclUCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config),
                "Error in ucc library configuration\n")
      ncclUCC_CHECK(ucc_init(&lib_params, lib_config, &lib_p),
                    "Error in ucc library initialization\n")
          ucc_lib_config_release(lib_config);

  ucc_context_params_t ctx_params = {
      .mask = UCC_CONTEXT_PARAM_FIELD_OOB,
      .oob.allgather = UCC_oob_allgather,
      .oob.req_test = UCC_oob_req_test,
      .oob.req_free = UCC_oob_req_free,
      .oob.coll_info = (void *)cComm,
      .oob.n_oob_eps = cComm->nranks,
      .oob.oob_ep = cComm->rank,
  };
  ucc_context_config_h ctx_config;
  ucc_context_h ctx;
  ncclUCC_CHECK(ucc_context_config_read(lib_p, NULL, &ctx_config),
                "Error in ucc context configuration\n")
      ncclUCC_CHECK(ucc_context_create(lib_p, &ctx_params, ctx_config, &ctx),
                    "Error in ucc context creation\n")
          ucc_context_config_release(ctx_config);

  ucc_team_params_t team_params = {
      .mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
              UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_TEAM_SIZE,
      .oob = ctx_params.oob,
      .ep = cComm->rank,
      .team_size = cComm->nranks,
      .ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG,
  };
  ucc_team_h team;
  ncclUCC_CHECK(
      ucc_team_create_post(&ctx, 1, &team_params, &team),
      "Error in ucc team creation") while (UCC_INPROGRESS ==
                                           ucc_team_create_test(team)) {
    ;
  };

  cComm->lib = lib_p;
  cComm->ctx = ctx;
  cComm->team = team;

  *collComm = cComm;
  return ncclSuccess;
}

ncclResult_t ncclUCCReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp,
                                  int *supported) {
  *supported = ((typeConvert(dataType) != -1) && (opConvert(redOp) != -1));
  return ncclSuccess;
}

ncclResult_t ncclUCCRegMr(void *collComm, void *data, int size, int type,
                          void **mhandle) {

  struct ncclUCCMemHandle *mh;
  NCCLCHECK(ncclIbMalloc((void **)&mh, sizeof(struct ncclUCCMemHandle)));
  mh->type = type;
  *mhandle = mh;
  return ncclSuccess;
}

ncclResult_t ncclUCCDeregMr(void *collComm, void *mhandle) {
  free(mhandle);
  return ncclSuccess;
}

ncclResult_t ncclUCCGetRequest(struct ncclUCCRequest *reqs,
                               struct ncclUCCRequest **req) {
  for (int i = 0; i < MAX_REQUESTS; i++) {
    struct ncclUCCRequest *r = reqs + i;
    if (r->used == 0) {
      r->used = 1;
      r->ucc_req = NULL;
      r->size = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("UCC : unable to allocate request");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t ncclUCCIallreduce(void *collComm, void *sendData, void *recvData,
                               int count, ncclDataType_t dataType,
                               ncclRedOp_t redOp, void *sendMhandle,
                               void *recvMhandle, void **request) {
  struct ncclUCCCollComm *cComm = (struct ncclUCCCollComm *)collComm;
  struct ncclUCCMemHandle *mr_src = (struct ncclUCCMemHandle *)sendMhandle;
  struct ncclUCCMemHandle *mr_dst = (struct ncclUCCMemHandle *)recvMhandle;

  ucc_coll_args_t args = {
      .mask = 0,
      .coll_type = UCC_COLL_TYPE_ALLREDUCE,
      .src.info.buffer = sendData,
      .src.info.count = count,
      .src.info.datatype = typeConvert(dataType),
      .src.info.mem_type =
          (mr_src->type == NCCL_PTR_CUDA ? UCC_MEMORY_TYPE_CUDA
                                         : UCC_MEMORY_TYPE_HOST),
      .dst.info.buffer = recvData,
      .dst.info.mem_type =
          (mr_dst->type == NCCL_PTR_CUDA ? UCC_MEMORY_TYPE_CUDA
                                         : UCC_MEMORY_TYPE_HOST),
      .op = opConvert(redOp),
  };
  args.dst.info.count = args.src.info.count;
  args.dst.info.datatype = args.src.info.datatype;
  ucc_coll_req_h ucc_req;
  ncclUCC_CHECK(ucc_collective_init(&args, &ucc_req, cComm->team),
                "Error during allreduce initialization");
  ncclUCC_CHECK(ucc_collective_post(ucc_req), "Error during allreduce post");

  struct ncclUCCRequest *req;
  NCCLCHECK(ncclUCCGetRequest(cComm->reqs, &req));

  req->requestType = NCCL_UCC_REQ_UCC_COLL;
  req->ucc_req = ucc_req;
  req->ctx = cComm->ctx;

  *request = req;

  return ncclSuccess;
}

// TODO
ncclResult_t ncclUCCIflush(void *collComm, void *data, int size, void *mhandle,
                           void **request) {
  return ncclSuccess;
}

ncclResult_t ncclUCCTest(void *request, int *done, int *size) {
  struct ncclUCCRequest *req = (struct ncclUCCRequest *)request;
  ncclUCC_CHECK(ucc_context_progress(req->ctx), "Error in context progress")
      ucc_status_t status = ucc_collective_test(req->ucc_req);

  if (status == UCC_OK) {
    *done = 1;
    ncclUCC_CHECK(ucc_collective_finalize(req->ucc_req),
                  "Error in collective finalization") req->used = 0;
  } else if (status == UCC_INPROGRESS) {
    *done = 0;
  } else {
    WARN("UCC: Error in collective test");
    return ncclInternalError;
  }

  return ncclSuccess;
}

ncclResult_t ncclUCCCloseColl(void *collComm) {
  struct ncclUCCCollComm *cComm = (struct ncclUCCCollComm *)collComm;

  ncclUCC_CHECK(ucc_team_destroy(cComm->team), "Error in team destruction")
      ncclUCC_CHECK(ucc_context_destroy(cComm->ctx),
                    "Error in context destruction")
          ncclUCC_CHECK(ucc_finalize(cComm->lib),
                        "Error in library finalization")

              NCCLCHECK(NCCL_PLUGIN_SYMBOL.closeRecv(cComm->recvComm));
  NCCLCHECK(NCCL_PLUGIN_SYMBOL.closeSend(cComm->sendComm));
  free(cComm);

  return ncclSuccess;
}

ncclResult_t ncclUCCCloseListen(void *listenComm) {
  struct ncclUCCListenComm *lComm = (struct ncclUCCListenComm *)listenComm;

  NCCLCHECK(NCCL_PLUGIN_SYMBOL.closeListen(lComm->listenCommP2P));
  free(listenComm);

  return ncclSuccess;
}

ncclCollNet_t uccPlugin = {
  "UCC",
  ncclUCCInit,
  ncclUCCDevices,
  ncclUCCGetProperties,
  ncclUCCListen,
  ncclUCCConnect,
  ncclUCCReduceSupport,
  ncclUCCRegMr,
  ncclUCCDeregMr,
  ncclUCCIallreduce,
  ncclUCCIflush,
  ncclUCCTest,
  ncclUCCCloseColl,
  ncclUCCCloseListen
};
