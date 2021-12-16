#include "vertexArrayObjects.hpp"


VAO::VAO(){
    glGenVertexArrays(1,&id_vao);
    glBindVertexArray(id_vao);
}
VAO::~VAO(){
    glDeleteBuffers(1,&id_vao);
}
void VAO::build(){
    glBindVertexArray(id_vao);
}
void VAO::release(){
    glBindVertexArray(0);
}