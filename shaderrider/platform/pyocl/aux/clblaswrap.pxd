from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport intptr_t, uintptr_t
from cpython cimport array
import array


#TODO move externs to separate pxd .. maybe clblas.pxd
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
        cl_float x
        cl_float y

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
    clblasStatus clblasCgemv(clblasOrder order,
                             clblasTranspose transA,
                             size_t M,
                             size_t N,
                             FloatComplex alpha,
                             const cl_mem A,
                             size_t offA,
                             size_t lda,
                             const cl_mem x,
                             size_t offx,
                             int incx,
                             FloatComplex beta,
                             cl_mem y,
                             size_t offy,
                             int incy,
                             cl_uint numCommandQueues,
                             cl_command_queue *commandQueues,
                             cl_uint numEventsInWaitList,
                             const cl_event *eventWaitList,
                             cl_event *events)
    clblasStatus clblasZgemv(clblasOrder order,
                             clblasTranspose transA,
                             size_t M,
                             size_t N,
                             FloatComplex alpha,
                             const cl_mem A,
                             size_t offA,
                             size_t lda,
                             const cl_mem x,
                             size_t offx,
                             int incx,
                             FloatComplex beta,
                             cl_mem y,
                             size_t offy,
                             int incy,
                             cl_uint numCommandQueues,
                             cl_command_queue *commandQueues,
                             cl_uint numEventsInWaitList,
                             const cl_event *eventWaitList,
                             cl_event *events)

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
    clblasStatus clblasCgemm(clblasOrder order,
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
    clblasStatus clblasZgemm(clblasOrder order,
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


cdef class EventList:
    cdef readonly intptr_t *data
    cdef readonly int n
    # FIXME def definicije ne mogu u pxd... pure python mora u pyx, a kako onda ovo?
    # def __cinit__(self, list events not None):
    #     cdef int i
    #     self.n = len(events)
    #     self.data = <intptr_t*> malloc(self.n*sizeof(intptr_t))
    #     if self.data == NULL:
    #         raise MemoryError('Unable to allocate memory for the EventList')
    #     for i in range(n):
    #         self.data[i] = <intptr_t>events[i].int_ptr
    # def __dealloc__(self):
    #     free(self.data)


cpdef setup()

cpdef teardown()

