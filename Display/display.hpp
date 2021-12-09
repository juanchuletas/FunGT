#if !defined(_DISPLAY_GL_H_)
#define _DISPLAY_GL_H_
#include<string>
#include<iostream>
#ifndef __APPLE__
int glewInit();
#endif
#ifdef __APPLE__
#define GLFW_INCLUDE_GLCOREARB
#include <OpenGL/gl3.h>
#include <OpenGL/gl3ext.h>
//#include <OpenGL/glu.h>
//#include <GLUT/glut.h>
#include <GLFW/glfw3.h>
#else
#ifdef _WIN32
  #include <windows.h>
#endif
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#endif
class Display{

    GLFWwindow *window;
    int width,height;
    std::string Windowname;

    public: 
      Display(int width, int height, std::string name);
      ~Display();
      // Methods
      int build();
};

#endif // _DISPLAY_GL_H_
