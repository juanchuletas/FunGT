#if !defined(_VERTEX_ARRAY_OBJECS_H_)
#define _VERTEX_ARRAY_OBJECS_H_
#include <GL/glew.h>

class VAO{
    unsigned int id_vao;


    public:
        VAO();
        ~VAO();
        void build();
        void release();
}; 

#endif // _VERTEX_ARRAY_OBJECS_H_
