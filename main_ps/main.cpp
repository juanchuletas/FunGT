#include "../funGT/fungt.hpp"

const unsigned int SCREEN_WIDTH = 1600;
const unsigned int SCREEN_HEIGHT = 1000;

int main(){
    std::string path = findProjectRoot();
    std::cout<<path<<std::endl;
    //Path to your shaders and models:

    std::string ps_vs = getAssetPath("resources/particle.vs");
    std::string ps_fs = getAssetPath("resources/particle.fs");
    
    //Creates a FunGT Scene to display 
    FunGTScene myGame = FunGT::createScene(SCREEN_WIDTH,SCREEN_HEIGHT);

    //Background color, use 255.f for pure white, 
    myGame->setBackgroundColor();

    //Initializes the Graphics Stuff
    myGame->initGL();

    //Gets an instance of the SceneManager class to render objects
    FunGTSceneManager scene_manager  = myGame->getSceneManager();

    //Shows infowindow:

    FunGTInfoWindow infowindow = myGame->getInfoWindow();
    flib::sycl_handler::get_device_info();
    std::shared_ptr<ParticleSimulation> pSys = std::make_shared<ParticleSimulation>(10000,ps_vs,ps_fs);
    

    myGame->set([&](){ // Sets up all the scenes in your game

        // Adds the renderable objects Animation and CubeMap to the SceneManager
        scene_manager->addRenderableObj(pSys);
    });

    myGame->render([&](){ // Renders the entire scene using data from the SceneManager
        scene_manager->renderScene();
        infowindow->renderGUI();
    });



    return 0;
}