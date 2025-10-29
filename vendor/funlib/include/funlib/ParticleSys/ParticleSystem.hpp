#if !defined(_PARTICLE_SYSTEM_H)
#define _PARTICLE_SYSTEM_H
#include <funlib/sycl/sycl_handler.hpp>
#include <vector>
#include <functional>
#include <cmath>
#include <funlib/ParticleSet/particle_set.hpp>
#include <CL/cl_gl.h>
namespace flib{
   
    template <typename T, typename Func>
    class ParticleSystem
    {
        class sycl_handler; // Forward declaration of sycl_handler
    public:
        void static update_ocl(int numOfParticles, unsigned int vbo, Func update_function, T dt){
            std::size_t n = static_cast<std::size_t>(numOfParticles);
            std::size_t xdim = static_cast<std::size_t>(std::ceil(std::sqrt(n)));
            std::size_t ydim = xdim; 
            // Get the SYCL queue from the sycl_handler   
            sycl::queue Q = flib::sycl_handler::get_queue();
            //OpeCL interop for updating particles
            cl_context clcontext = flib::sycl_handler::get_clContext();
                
            cl_command_queue clqueue = sycl::get_native<sycl::backend::opencl>(Q);

            cl_mem clbuffer = clCreateFromGLBuffer(clcontext, CL_MEM_READ_WRITE, vbo, NULL);
            if (clbuffer == NULL) {
                throw std::runtime_error("Failed to create OpenCL buffer from GL buffer");
            }
            sycl::context syclCtx = flib::sycl_handler::get_sycl_context();
            
            clEnqueueAcquireGLObjects(clqueue, 1, &clbuffer, 0, NULL, NULL);    
            {
                sycl::buffer<Particle<T>> buf =
                sycl::make_buffer<sycl::backend::opencl, Particle<T>>(clbuffer, syclCtx);

                Q.submit([&](sycl::handler &cgh) {
                    auto acc = buf.template get_access<sycl::access::mode::read_write>(cgh);
                    cgh.parallel_for(sycl::range<2>(sycl::range<2>{static_cast<size_t>(xdim),static_cast<size_t>(ydim)}),
                        [=](sycl::item<2> item){
                             //user defined lambda function
                            std::size_t index = item[0] * ydim + item[1];
                            if (index < n) {
                                update_function(acc[index], dt);
                            }
                    });

                });

                 Q.wait();

                  // FORCE SYNC: Just create host accessor to trigger sync, but don't copy
                {
                    auto host_acc = buf.template get_access<sycl::access::mode::read>();
                    // Just creating the accessor forces the sync back to OpenCL buffer
                    // No need to actually use host_acc or copy data
                    (void)host_acc; // Suppress unused variable warning
                }
            }
             clFinish(clqueue);
            //Release OpenCL buffer
            clEnqueueReleaseGLObjects(clqueue, 1, &clbuffer, 0, NULL, NULL);
            clFinish(clqueue);

            glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
            clReleaseMemObject(clbuffer);
        }
    public:
    
        void static update(ParticleSet<T> &particles, Func update_function, T dt){

            std::size_t n = particles._particles.size();
            std::size_t xdim = static_cast<std::size_t>(std::ceil(std::sqrt(n)));
            std::size_t ydim = xdim;    
            sycl::queue Q = flib::sycl_handler::get_queue();
            {
               
                sycl::buffer<Particle<T>> buf(particles._particles.data(), sycl::range<1>(n));
                Q.submit([&](sycl::handler &cgh) {
                auto acc = buf.template get_access<sycl::access::mode::read_write>(cgh);
                    cgh.parallel_for(sycl::range<2>(sycl::range<2> {static_cast<size_t>(xdim),static_cast<size_t>(ydim)}),[=](sycl::item<2> item) {
                    //user defined lambda function
                    std::size_t index = item[0] * ydim + item[1];
                    if (index < n) {
                        update_function(acc[index], dt);
                    }
                    });
                 });

            }

        }
    
    };
    
}

#endif // _PARTICLE_SYSTEM_H
