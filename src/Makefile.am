#
# Copyright (c) 2019-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

lib_LTLIBRARIES     = libnccl-net.la

libnccl_net_la_CPPFLAGS = -I$(top_srcdir)/include
libnccl_net_la_CFLAGS   = $(CFLAGS) -DGNU_SOURCE
libnccl_net_la_LIBADD   = -lcudart_static
libnccl_net_la_LDFLAGS  = $(LDFLAGS)

libnccl_net_la_SOURCES = \
	ibvwrap.c \
	utils.c	\
        param.c \
	socket.c \
	p2p_plugin.c \
	ib_plugin.c

if HAVE_UCX_PLUGIN
libnccl_net_la_CPPFLAGS += -DHAVE_UCX_PLUGIN $(UCX_CPPFLAGS)
libnccl_net_la_LIBADD   += $(UCX_LIBADD)
libnccl_net_la_LDFLAGS  += $(UCX_LDFLAGS)
libnccl_net_la_SOURCES  += \
	ucx_plugin.c \
	ucx_rma_plugin.c
endif

if HAVE_SHARP_PLUGIN
libnccl_net_la_CPPFLAGS += -DHAVE_SHARP_PLUGIN $(SHARP_CPPFLAGS)
libnccl_net_la_LIBADD   += $(SHARP_LIBADD)
libnccl_net_la_LDFLAGS  += $(SHARP_LDFLAGS)
libnccl_net_la_SOURCES  += sharp_plugin.c
endif

if HAVE_UCC_PLUGIN
libnccl_net_la_CPPFLAGS += -DHAVE_UCC_PLUGIN $(UCC_CPPFLAGS)
libnccl_net_la_LIBADD   += $(UCC_LIBADD)
libnccl_net_la_LDFLAGS  += $(UCC_LDFLAGS)
libnccl_net_la_SOURCES  += ucc_plugin.c
endif
