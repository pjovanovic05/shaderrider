"""
Wraps clBLAS minimally to support shaderrider.

Based on Eric Hunsberger's clblass wrapper, but with
support for more than one event in the wait_for lists.
"""

import atexit
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport intptr_t, uintptr_t


cdef extern from "clBLAS.h":
    ctypedef enum clblasStatus:
        clblasSuccess
        clblasInvalidValue
        clblasInvalidCommandQueue
        clblasInvalidContext
        clblasInvalidMemObject
        clblasInvalidDevice
        clblasInvalidEventWaitList
        clblasOutOfResources
        clblasOutOfHostMemory
        clblasInvalidOperation
        clblasCompilerNotAvailable
        clblasBuildProgramFailure
        clblasNotImplemented
        clblasNotInitialized
        clblasInvalidMatA
        clblasInvalidMatB
        clblasInvalidMatC
        clblasInvalidVecX
        clblasInvalidVecY
        clblasInvalidDim
        clblasInvalidLeadDimA
        clblasInvalidLeadDimB
        clblasInvalidLeadDimC
        clblasInvalidIncX
        clblasInvalidIncY
        clblasInsufficientMemMatA
        clblasInsufficientMemMatB
        clblasInsufficientMemMatC
        clblasInsufficientMemVecX
        clblasInsufficientMemVecY

    ctypedef float cl_float
    ctypedef double cl_double
    ctypedef unsigned int cl_uint

    ctypedef struct cl_float2:
        cl_float x
        cl_float y

    ctypedef struct cl_double2:
        cl_double x
        cl_double y

    struct _cl_mem:
        pass

    struct _cl_command_queue:
        pass

    struct _cl_event:
        pass

    ctypedef _cl_mem* cl_mem
    ctypedef _cl_command_queue* cl_command_queue
    ctypedef _cl_event* cl_event

    ctypedef enum clblasOrder:
        clblasRowMajor
        clblasColumnMajor

    ctypedef enum clblasTranspose:
        clblasNoTrans
        clblasTrans
        clblasConjTrans

    clblasStatus clblasSetup()
    void clblasTeardown()

    clblasStatus clblasSgemv(clblasOrder order,
                             clblasTranspose transA,
                             size_t M,
                             size_t N,
                             cl_float alpha,
                             const cl_mem A,
                             size_t offA,
                             size_t lda,
                             const cl_mem x,
                             size_t offx,
                             int incx,
                             cl_float beta,
                             cl_mem y,
                             size_t offy,
                             int incy,
                             cl_uint numCommandQueues,
                             cl_command_queue *commandQueues,
                             cl_uint numEventsInWaitList,
                             const cl_event *eventWaitList,
                             cl_event *events)
    clblasStatus clblasDgemv(clblasOrder order,
                             clblasTranspose transA,
                             size_t M,
                             size_t N,
                             cl_double alpha,
                             const cl_mem A,
                             size_t offA,
                             size_t lda,
                             const cl_mem x,
                             size_t offx,
                             int incx,
                             cl_double beta,
                             cl_mem y,
                             size_t offy,
                             int incy,
                             cl_uint numCommandQueues,
                             cl_command_queue *commandQueues,
                             cl_uint numEventsInWaitList,
                             const cl_event *eventWaitList,
                             cl_event *events)
    #clblasStatus clblasCgemv(clblasOrder order,
    #                         clblasTranspose transA,
    #                         size_t M,
    #                         size_t N,
    #                         cl_float2 alpha,
    #                         const cl_mem A,
    #                         size_t offA,
    #                         size_t lda,
    #                         const cl_mem x,
    #                         size_t offx,
    #                         int incx,
    #                         cl_float2 beta,
    #                         cl_mem y,
    #                         size_t offy,
    #                         int incy,
    #                         cl_uint numCommandQueues,
    #                         cl_command_queue *commandQueues,
    #                         cl_uint numEventsInWaitList,
    #                         const cl_event *eventWaitList,
    #                         cl_event *events)
    #clblasStatus clblasZgemv(clblasOrder order,
    #                         clblasTranspose transA,
    #                         size_t M,
    #                         size_t N,
    #                         cl_double2 alpha,
    #                         const cl_mem A,
    #                         size_t offA,
    #                         size_t lda,
    #                         const cl_mem x,
    #                         size_t offx,
    #                         int incx,
    #                         cl_double2 beta,
    #                         cl_mem y,
    #                         size_t offy,
    #                         int incy,
    #                         cl_uint numCommandQueues,
    #                         cl_command_queue *commandQueues,
    #                         cl_uint numEventsInWaitList,
    #                         const cl_event *eventWaitList,
    #                         cl_event *events)

    clblasStatus clblasSger(clblasOrder order,
                            size_t M,
                            size_t N,
                            cl_float alpha,
                            const cl_mem X,
                            size_t offx,
                            int incx,
                            const cl_mem Y,
                            size_t offy,
                            int incy,
                            cl_mem A,
                            size_t offa,
                            size_t lda,
                            cl_uint numCommandQueues,
                            cl_command_queue *commandQueues,
                            cl_uint numEventsInWaitList,
                            const cl_event *eventWaitList,
                            cl_event *events)
    clblasStatus clblasDger(clblasOrder order,
                            size_t M,
                            size_t N,
                            cl_double alpha,
                            const cl_mem X,
                            size_t offx,
                            int incx,
                            const cl_mem Y,
                            size_t offy,
                            int incy,
                            cl_mem A,
                            size_t offa,
                            size_t lda,
                            cl_uint numCommandQueues,
                            cl_command_queue *commandQueues,
                            cl_uint numEventsInWaitList,
                            const cl_event *eventWaitList,
                            cl_event *events)

    clblasStatus clblasSgemm(clblasOrder order,
                             clblasTranspose transA,
                             clblasTranspose transB,
                             size_t M,
                             size_t N,
                             size_t K,
                             cl_float alpha,
                             const cl_mem A,
                             size_t offA,
                             size_t lda,
                             const cl_mem B,
                             size_t offB,
                             size_t ldb,
                             cl_float beta,
                             cl_mem C,
                             size_t offC,
                             size_t ldc,
                             cl_uint numCommandQueues,
                             cl_command_queue *commandQueues,
                             cl_uint numEventsInWaitList,
                             const cl_event *eventWaitList,
                             cl_event *events)
    clblasStatus clblasDgemm(clblasOrder order,
                             clblasTranspose transA,
                             clblasTranspose transB,
                             size_t M,
                             size_t N,
                             size_t K,
                             cl_float alpha,
                             const cl_mem A,
                             size_t offA,
                             size_t lda,
                             const cl_mem B,
                             size_t offB,
                             size_t ldb,
                             cl_float beta,
                             cl_mem C,
                             size_t offC,
                             size_t ldc,
                             cl_uint numCommandQueues,
                             cl_command_queue *commandQueues,
                             cl_uint numEventsInWaitList,
                             const cl_event *eventWaitList,
                             cl_event *events)
    #clblasStatus clblasCgemm(clblasOrder order,
    #                         clblasTranspose transA,
    #                         clblasTranspose transB,
    #                         size_t M,
    #                         size_t N,
    #                         size_t K,
    #                         cl_float2 alpha,
    #                         const cl_mem A,
    #                         size_t offA,
    #                         size_t lda,
    #                         const cl_mem B,
    #                         size_t offB,
    #                         size_t ldb,
    #                         cl_float2 beta,
    #                         cl_mem C,
    #                         size_t offC,
    #                         size_t ldc,
    #                         cl_uint numCommandQueues,
    #                         cl_command_queue *commandQueues,
    #                         cl_uint numEventsInWaitList,
    #                         const cl_event *eventWaitList,
    #                         cl_event *events)
    #clblasStatus clblasZgemm(clblasOrder order,
    #                         clblasTranspose transA,
    #                         clblasTranspose transB,
    #                         size_t M,
    #                         size_t N,
    #                         size_t K,
    #                         cl_double2 alpha,
    #                         const cl_mem A,
    #                         size_t offA,
    #                         size_t lda,
    #                         const cl_mem B,
    #                         size_t offB,
    #                         size_t ldb,
    #                         cl_double2 beta,
    #                         cl_mem C,
    #                         size_t offC,
    #                         size_t ldc,
    #                         cl_uint numCommandQueues,
    #                         cl_command_queue *commandQueues,
    #                         cl_uint numEventsInWaitList,
    #                         const cl_event *eventWaitList,
    #                         cl_event *events)


cdef get_status_message(clblasStatus status):
    if status == clblasSuccess:
        return "success"
    if status == clblasInvalidValue:
        return "invalid value"
    if status == clblasInvalidCommandQueue:
        return "invalid command queue"
    if status == clblasInvalidContext:
        return "invalid context"
    if status == clblasInvalidMemObject:
        return "invalid mem object"
    if status == clblasInvalidDevice:
        return "invalid device"
    if status == clblasInvalidEventWaitList:
        return "invalid event wait list"
    if status == clblasOutOfResources:
        return "out of resources"
    if status == clblasOutOfHostMemory:
        return "out of host memory"
    if status == clblasInvalidOperation:
        return "invalid operation"
    if status == clblasCompilerNotAvailable:
        return "compiler not available"
    if status == clblasBuildProgramFailure:
        return "build program failure"
    if status == clblasNotImplemented:
        return "clBLAS: not implemented"
    if status == clblasNotInitialized:
        return "clBLAS: not initialized"
    if status == clblasInvalidMatA:
        return "clBLAS: invalid mat A"
    if status == clblasInvalidMatB:
        return "clBLAS: invalid mat B"
    if status == clblasInvalidMatC:
        return "clBLAS: invalid mat C"
    if status == clblasInvalidVecX:
        return "clBLAS: invalid vec X"
    if status == clblasInvalidVecY:
        return "clBLAS: invalid vec Y"
    if status == clblasInvalidDim:
        return "clBLAS: invalid dim"
    if status == clblasInvalidLeadDimA:
        return "clBLAS: invalid lead dim A"
    if status == clblasInvalidLeadDimB:
        return "clBLAS: invalid lead dim B"
    if status == clblasInvalidLeadDimC:
        return "clBLAS: invalid lead dim C"
    if status == clblasInvalidIncX:
        return "clBLAS: invalid inc X"
    if status == clblasInvalidIncY:
        return "clBLAS: invalid inc Y"
    if status == clblasInsufficientMemMatA:
        return "clBLAS: insufficient mem mat A"
    if status == clblasInsufficientMemMatB:
        return "clBLAS: insufficient mem mat B"
    if status == clblasInsufficientMemMatC:
        return "clBLAS: insufficient mem mat C"
    if status == clblasInsufficientMemVecX:
        return "clBLAS: insufficient mem vec X"
    if status == clblasInsufficientMemVecY:
        return "clBLAS: insufficient mem vec Y"
    return "unrecognized status (code %d)" % status


dtype_size = {
    np.dtype('float32'): 4,
    np.dtype('float64'): 8,
    np.dtype('complex64'): 8,
    np.dtype('complex128'): 16
}


cpdef dtypes_str(dtypes):
    if len(dtypes) == 1:
        return "'%s'" % dtypes[0]
    return 'one of %s' % dtypes


cpdef check_dtype(args, dtypes):
    dtype = args[0].dtype
    if not all([arg.dtype == dtype for arg in args]):
        raise ValueError('All arguments must have the same dtype (%s)' % dtypes_str(dtype))
    if dtype not in dtypes:
        raise ValueError('Data type must be %s' % dtypes_str(dtypes))
    return dtype


cpdef check_array(a, int ndim, str name):
    if not isinstance(a, clarray.Array):
        raise ValueError("'%s' must be a PyOpenCL Array" % name)
    if not len(a.shape) == ndim:
        raise ValueError("'%s' must have %d dimensions (got %d)" % (name, ndim, len(a.shape)))


cpdef check_matrix(a, str name):
    check_array(a, 2, name)


cpdef check_vector(a, str name):
    check_array(a, 1, name)


cpdef check_shape_dim(shape, size_t dim, size_t target, str name):
    if shape[dim] != target:
        raise ValueError("'%s.shape[%d]' must be %d (got %d)" % (name, dim, target, shape[dim]))


cpdef setup():
    """Setup the clBLAS library"""
    global is_setup
    cdef clblasStatus err
    if not is_setup:
        err = clblasSetup()
        if err != clblasSuccess:
            raise RuntimeError('Failed to setup clBLAS (Error %d)' % err)
        else:
            is_setup = True


cpdef teardown():
    """Teardown the clBLAS library (called automatically at exit)"""
    global is_setup
    if is_setup:
        clblasTeardown()
        is_setup = False


is_setup = False
atexit.register(teardown)   # TODO do we really need this?


def ger(queue, A, x, y, float alpha=1.0, clblasOrder order=clblasRowMajor, list wait_for=None):
    """A <- alpha*X*Y^T + A"""
    dtype = check_dtype([A, x, y], ['float32', 'float64'])
    check_matrix(A, 'A')
    check_vector(x, 'x')
    check_vector(y, 'y')

    cdef size_t M = A.shape[0]
    cdef size_t N = A.shape[1]
    check_shape_dim(x.shape, 0, M, 'x')
    check_shape_dim(y.shape, 0, N, 'y')

    cdef size_t element_size = dtype_size[dtype]
    cdef cl_mem Adata = <cl_mem><intptr_t>A.base_data.int_ptr
    cdef size_t offA = A.offset / element_size
    cdef size_t lda = A.strides[0] / element_size
    cdef cl_mem xdata = <cl_mem><intptr_t>x.base_data.int_ptr
    cdef size_t offx = x.offset / element_size
    cdef size_t incx = x.strides[0] / element_size
    cdef cl_mem ydata = <cl_mem><intptr_t>y.base_data.int_ptr
    cdef size_t offy = y.offset / element_size
    cdef size_t incy = y.strides[0] / element_size

    cdef cl_command_queue commandQueue = <cl_command_queue><intptr_t>queue.int_ptr
    cdef EventList el = None if wait_for is None else EventList(wait_for)
    cdef cl_event myevent = NULL

    cdef clblasStatus err = clblasSuccess

    if dtype == np.dtype('float32'):
        err = clblasSger(order, M, N,
                         <cl_float>alpha, xdata, offx, incx,
                         ydata, offy, incy,
                         Adata, offA, lda,
                         1, &commandQueue,
                         0 if el is None else el.n,
                         NULL if el is None else <cl_event*>el.data,
                         &myevent)
    elif dtype == np.dtype('float64'):
        err = clblasDger(order, M, N,
                         <cl_double>alpha, xdata, offx, incx,
                         ydata, offy, incy,
                         Adata, offA, lda,
                         1, &commandQueue,
                         0 if el is None else el.n,
                         NULL if el is None else <cl_event*>el.data,
                         &myevent)
    else:
        raise ValueError("Unrecognized dtype '%s'" % dtype)

    if err != clblasSuccess:
        raise RuntimeError("'ger' failed: %s" % get_status_message(err))

    return cl.Event.from_int_ptr(<intptr_t>myevent)


def gemv(queue, A, x, y, transA=False, float alpha=1.0, float beta=1.0,
         clblasOrder order=clblasRowMajor, list wait_for=None):
    """y <- alpha*dot(A,x) + beta*y"""
    dtype = check_dtype([A, x, y], ['float32', 'float64', 'complex64', 'complex128'])
    check_matrix(A, 'A')
    check_vector(x, 'x')
    check_vector(y, 'y')

    cdef size_t M = A.shape[0]
    cdef size_t N = A.shape[1]
    check_shape_dim(x.shape, 0, M if transA else N, 'x')
    check_shape_dim(y.shape, 0, N if transA else M, 'y')

    cdef size_t element_size = dtype_size[dtype]
    cdef cl_mem Adata = <cl_mem><intptr_t>A.base_data.int_ptr
    cdef size_t offA = A.offset / element_size
    cdef size_t lda = A.strides[0] / element_size
    cdef cl_mem xdata = <cl_mem><intptr_t>x.base_data.int_ptr
    cdef size_t offx = x.offset / element_size
    cdef size_t incx = x.strides[0] / element_size
    cdef cl_mem ydata = <cl_mem><intptr_t>y.base_data.int_ptr
    cdef size_t offy = y.offset / element_size
    cdef size_t incy = y.strides[0] / element_size

    cdef cl_command_queue commandQueue = <cl_command_queue><intptr_t>queue.int_ptr
    cdef EventList el = None if wait_for is None else EventList(wait_for)
    cdef cl_event myevent = NULL

    cdef clblasStatus err = clblasSuccess

    if dtype == np.dtype('float32'):
        err = clblasSgemv(order,
                          clblasTrans if transA else clblasNoTrans,
                          M, N, <cl_float>alpha, Adata, offA, lda,
                          xdata, offx, incx, <cl_float>beta, ydata, offy, incy,
                          1, &commandQueue,
                          0 if el is None else el.n,
                          NULL if el is None else <cl_event*>el.data,
                          &myevent)
    elif dtype == np.dtype('float64'):
        err = clblasDgemv(order,
                          clblasTrans if transA else clblasNoTrans,
                          M, N, <cl_double>alpha, Adata, offA, lda,
                          xdata, offx, incx, <cl_double>beta, ydata, offy, incy,
                          1, &commandQueue,
                          0 if el is None else el.n,
                          NULL if el is None else <cl_event*>el.data,
                          &myevent)
    #elif dtype == np.dtype('complex64'):
    #    err = clblasCgemv(order,
    #                      clblasTrans if transA else clblasNoTrans,
    #                      M, N, <cl_float2>alpha, Adata, offA, lda,
    #                      xdata, offx, incx, <cl_float2>beta, ydata, offy, incy,
    #                      1, &commandQueue,
    #                      0 if el is None else el.n,
    #                      NULL if el is None else <cl_event*>el.data,
    #                      &myevent)
    #elif dtype == np.dtype('complex128'):
    #    err = clblasZgemv(order,
    #                      clblasTrans if transA else clblasNoTrans,
    #                      M, N, <cl_double2>alpha, Adata, offA, lda,
    #                      xdata, offx, incx, <cl_double2>beta, ydata, offy, incy,
    #                      1, &commandQueue,
    #                      0 if el is None else el.n,
    #                      NULL if el is None else <cl_event*>el.data,
    #                      &myevent)
    else:
        raise ValueError("Unrecognized dtype '%s'" % dtype)

    if err != clblasSuccess:
        raise RuntimeError("'gemv' failed: %s" % get_status_message(err))

    return cl.Event.from_int_ptr(<intptr_t>myevent)


# TODO check if lda, ldb and ldc are calculated correctly with regards to order!!
def gemm(queue, A, B, C, transA=False, transB=False, float alpha=1.0, float beta=0.0,
           clblasOrder order=clblasRowMajor, list wait_for=None):
    """C <- alpha*dot(A, B) + beta*C"""
    dtype = check_dtype([A, B, C], ['float32', 'float64', 'complex64', 'complex128'])
    check_matrix(A, 'A')
    check_matrix(B, 'B')
    check_matrix(C, 'C')

    cdef size_t M = A.shape[1 if transA else 0]
    cdef size_t K = A.shape[0 if transA else 1]
    cdef size_t N = B.shape[0 if transB else 1]
    check_shape_dim(B.shape, 1 if transB else 0, K, 'B')
    check_shape_dim(C.shape, 0, M, 'C')
    check_shape_dim(C.shape, 1, N, 'C')

    cdef size_t element_size = dtype_size[dtype]
    cdef cl_mem Adata = <cl_mem><intptr_t>A.base_data.int_ptr
    cdef size_t offA = A.offset / element_size
    cdef size_t lda = A.strides[0] / element_size
    cdef cl_mem Bdata = <cl_mem><intptr_t>B.base_data.int_ptr
    cdef size_t offB = B.offset / element_size
    cdef ldb = B.strides[0] / element_size
    cdef cl_mem Cdata = <cl_mem><intptr_t>C.base_data.int_ptr
    cdef size_t offC = C.offset / element_size
    cdef size_t ldc = C.strides[0] / element_size

    cdef cl_command_queue commandQueue = <cl_command_queue><intptr_t>queue.int_ptr
    cdef EventList el = EventList(wait_for) if not (wait_for is None) else None
    cdef cl_event myevent = NULL

    cdef clblasStatus err = clblasSuccess

    if dtype == np.dtype('float32'):
        err = clblasSgemm(order,
                          clblasTrans if transA else clblasNoTrans,
                          clblasTrans if transB else clblasNoTrans,
                          M, N, K,
                          <cl_float>alpha, Adata, offA, lda, Bdata, offB, ldb,
                          <cl_float>beta, Cdata, offC, ldc,
                          1, &commandQueue,
                          0 if el is None else el.n,
                          NULL if el is None else <cl_event*>el.data,
                          &myevent)
    elif dtype == np.dtype('float64'):
        err = clblasDgemm(order,
                          clblasTrans if transA else clblasNoTrans,
                          clblasTrans if transB else clblasNoTrans,
                          M, N, K,
                          <cl_double>alpha, Adata, offA, lda, Bdata, offB, ldb,
                          <cl_double>beta, Cdata, offC, ldc,
                          1, &commandQueue,
                          0 if el is None else el.n,
                          NULL if el is None else <cl_event*>el.data,
                          &myevent)
    #elif dtype == np.dtype('complex64'):
    #    err = clblasCgemm(order,
    #                      clblasTrans if transA else clblasNoTrans,
    #                      clblasTrans if transB else clblasNoTrans,
    #                      M, N, K,
    #                      <cl_float2>cl_float2(x=alpha.real, y=alpha.imag),
    #                      Adata, offA, lda, Bdata, offB, ldb,
    #                      <cl_float2>cl_float2(x=beta.real, y=beta.imag),
    #                      Cdata, offC, ldc,
    #                      1, &commandQueue,
    #                      0 if el is None else el.n,
    #                      NULL if el is None else <cl_event*>el.data,
    #                      &myevent)
    #elif dtype == np.dtype('complex128'):
    #    err = clblasZgemm(order,
    #                      clblasTrans if transA else clblasNoTrans,
    #                      clblasTrans if transB else clblasNoTrans,
    #                      M, N, K,
    #                      <cl_double2>cl_double2(x=alpha.real, y=alpha.imag),
    #                      Adata, offA, lda, Bdata, offB, ldb,
    #                      <cl_double2>cl_double2(x=beta.real, y=beta.imag),
    #                      Cdata, offC, ldc,
    #                      1, &commandQueue,
    #                      0 if el is None else el.n,
    #                      NULL if el is None else <cl_event*>el.data,
    #                      &myevent)
    else:
        raise ValueError("Unrecognized dtype '%s'" % dtype)

    if err != clblasSuccess:
        raise RuntimeError("'gemm' failed: %s" % get_status_message(err))

    return cl.Event.from_int_ptr(<intptr_t>myevent)


cdef class EventList:
    cdef intptr_t *data
    cdef readonly int n

    def __cinit__(self, list events not None):
        cdef int i
        self.n = len(events)
        self.data = <intptr_t*> malloc(self.n*sizeof(intptr_t))
        if self.data == NULL:
            raise MemoryError('Unable to allocate memory for the EventList')
        for i in range(self.n):
            self.data[i] = <intptr_t>events[i].int_ptr

    def __dealloc__(self):
        if self.data != NULL:
            free(self.data)

