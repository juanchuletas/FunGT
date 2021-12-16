#if !defined(_VERTEX_INDICES_H_)
#define _VERTEX_INDICES_H_
#include <GL/glew.h>

class VI{

    unsigned int id_rnd;
    unsigned int numId_rnd;

    public:
        VI(const unsigned int *data, unsigned int totIndices);
        ~VI();


        void build();
        void release();
        unsigned int getNumIndices() const ; 

};

#endif // _VERTEX_INDICES_H_
