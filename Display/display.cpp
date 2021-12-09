#include"display.hpp"



Display::Display(int _width, int _height, std::string _name)
:   width{_width}, height{_height}, Windowname{_name}{
        std::cout<<"Display constructor\n";
        
}
Display::~Display(){
    std::cout<<"Display destructor\n";
    // Delete window before ending the program
    glfwDestroyWindow(window);
    // Terminate GLFW before ending the program
    glfwTerminate();
}

// METHODS
int Display::build(){

    if (!glfwInit())
        return -1;
        // Tell GLFW what version of OpenGL we are using 
        // In this case we are using OpenGL 3.3
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,4);
    glfwWindowHint(GLFW_RESIZABLE,GL_FALSE);

        // Tell GLFW we are using the CORE profile
        // So that means we only have the modern functions
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(width,height,Windowname.c_str(), NULL, NULL);
    if (!window)
    {
         glfwTerminate();
        return -1;
    }
        // Introduce the window into the current context
    glfwMakeContextCurrent(window);
        // Specify the viewport of OpenGL in the Window
        // In this case the viewport goes from x = 0, y = 0, to x = widtgh, y = height
    glViewport(0,0,width,height);
         // Specify the color of the background
    glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
        // Clean the back buffer and assign the new color to it
    
    if(glfwInit()!=GLFW_TRUE)
    {
            std::cout<<"ERROR"<<std::endl;
    }
    //The window only vanishes with the close event c
    while (!glfwWindowShouldClose(window))
    {
        //Render
         glClear(GL_COLOR_BUFFER_BIT);
        
        //glUseProgram(shaderProgram);
        //glBindVertexArray(VAO);
        //glDrawArrays(GL_TRIANGLES,0,3);
        
        // Swap the back buffer with the front buffer
        glfwSwapBuffers(window);
        /* Poll for and process events */
        glfwPollEvents();
    }

    return 0;
}
