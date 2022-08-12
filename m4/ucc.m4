#
# Copyright (c) 2001-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_UCC],[

AS_IF([test "x$ucc_checked" != "xyes"],[

    ucc_happy="no"

    AC_ARG_WITH([ucc],
            [AS_HELP_STRING([--with-ucc=(DIR)], [Enable the use of UCC (default is guess).])],
            [], [with_ucc=guess])

    AS_IF([test "x$with_ucc" != "xno"],
    [
        save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"

        AS_IF([test ! -z "$with_ucc" -a "x$with_ucc" != "xyes" -a "x$with_ucc" != "xguess"],
        [
            check_ucc_dir="$with_ucc"
            check_ucc_libdir="$with_ucc/lib"
            CPPFLAGS="-I$with_ucc/include $save_CPPFLAGS"
            LDFLAGS="-L$check_ucc_libdir $save_LDFLAGS"
        ])

        AS_IF([test "x$check_ucc_dir" = "x" -a "x$HPCX_UCC_DIR" != "x"],
        [
            check_ucc_dir="$HPCX_UCC_DIR"
            check_ucc_libdir="$HPCX_UCC_DIR/lib"
            CPPFLAGS="-I$check_ucc_dir/include $save_CPPFLAGS"
            LDFLAGS="-L$check_ucc_libdir $save_LDFLAGS"
        ])

        AS_IF([test ! -z "$with_ucc_libdir" -a "x$with_ucc_libdir" != "xyes"],
        [
            check_ucc_libdir="$with_ucc_libdir"
            LDFLAGS="-L$check_ucc_libdir $save_LDFLAGS"
        ])

        AC_CHECK_HEADERS([ucc/api/ucc.h],
        [
            AC_CHECK_LIB([ucc], [ucc_collective_post],
            [
                ucc_happy="yes"
            ],
            [
                ucc_happy="no"
            ], [-lucc])
        ],
        [
            ucc_happy="no"
        ])

        AS_IF([test "x$ucc_happy" = "xyes"],
        [
            AS_IF([test "x$check_ucc_dir" != "x"],
            [
                AC_MSG_RESULT([UCC dir: $check_ucc_dir])
                AC_SUBST(UCC_CPPFLAGS, "-I$check_ucc_dir/include/")
            ])

            AS_IF([test "x$check_ucc_libdir" != "x"],
            [
                AC_SUBST(UCC_LDFLAGS, "-L$check_ucc_libdir")
            ])

            AC_SUBST(UCX_LIBADD, "-lucc")
        ],
        [
            AS_IF([test "x$with_ucc" != "xguess"],
            [
                AC_MSG_ERROR([UCC support is requested but UCC package cannot be found])
            ],
            [
                AC_MSG_WARN([UCC not found])
            ])
        ])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

    ],
    [
        AC_MSG_WARN([UCC was explicitly disabled])
    ])

    ucc_checked=yes
    AM_CONDITIONAL([HAVE_UCC_PLUGIN], [test "x$ucc_happy" != xno])
])

])
