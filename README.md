# shaderrider

Shaderrider is a python package that enables construction of neural networks using expression graphs, and their execution on OpenCL accelerators. 

The main goal of this framework was to explore low level implementation details involved in creating frameworks such as Theano, PyTorch, MXNet and similar, but also to create the simplest possible implementation, with as little concepts as neccessary, which would foster easy extensibility.

The project utilizes PyOpenCL, Cython and clBLAS.
