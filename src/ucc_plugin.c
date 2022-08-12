/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "config.h"
#include "core.h"
#include "nccl.h"
#include "nccl_net.h"
#include "p2p_plugin.h"
#include "param.h"
#include "utils.h"

ncclResult_t ncclUCCInit(ncclDebugLogger_t logFunction) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCDevices(int* ndev) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCGetProperties(int dev, ncclNetProperties_t* props) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCListen(int dev, void* opaqueHandle, void** listenComm) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCConnect(void* handles[], int nranks, int rank,
                              void* listenComm, void** collComm) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp,
                                  int* supported) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCRegMr(void* collComm, void* data, int size, int type,
                          void** mhandle) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCDeregMr(void* collComm, void* mhandle) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCIallreduce(void* collComm, void* sendData, void* recvData,
                               int count, ncclDataType_t dataType,
                               ncclRedOp_t redOp, void* sendMhandle,
                               void* recvMhandle, void** request) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCIflush(void* collComm, void* data, int size, void* mhandle,
                           void **request) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCTest(void* request, int* done, int* size) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCCloseColl(void* collComm) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
}

ncclResult_t ncclUCCCloseListen(void* listenComm) {
  WARN("UCC plugin is not implemented");
  return ncclInternalError;
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
