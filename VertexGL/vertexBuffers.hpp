#if !defined(_VERTEX_BUFFER_H_)
#define _VERTEX_BUFFER_H_
#include <GL/glew.h>
class VB{

    private: 
        unsigned int id_Render; // OpenGL  needs a numeric ID, keeps track of every type of object: texture, vertex, shader, etc. 
    public:
        VB(const void* data, unsigned int size);
        ~VB();


        void build();
        void release();


};


#endif // _VERTEX_BUFFER_H_
