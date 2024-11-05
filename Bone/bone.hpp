#if !defined(_BONE_H_)
#define _BONE_H_
#include <vector>
#include <string>
#include <assimp/anim.h>
#include "../include/glmath.hpp"
#include "../Helpers/helpers.hpp"
#include "../Matrix/matrix4x4f.h"
#include <glm/gtx/quaternion.hpp>
struct KeyPosition{

    aiVector3D m_position; 
    float m_timeStamp; 
};
struct KeyRotation{

    aiQuaternion m_rotation; 
    float m_timeStamp;

};
struct KeyScale{

    aiVector3D m_scale; 
    float m_timeStamp; 

};

class Bone{


    private:
        std::vector<KeyPosition> m_bPosition; 
        std::vector<KeyRotation> m_bRotation; 
        std::vector<KeyScale>    m_bScale; 
        int m_numPos; 
        int m_numRot; 
        int m_numScale;

        glm::mat4 m_bLocaleTrans;
        fungl::Matrix4f m_BoneLocalTransformsMat;
        std::string m_bName; 
        int m_bID;

    public: 

        Bone(const std::string &name, int id, const aiNodeAnim* channel);
        ~Bone();
        //Setters
        void setBonePos(const aiNodeAnim *channel); 
        void setBoneRot(const aiNodeAnim *channel); 
        void setBoneScale(const aiNodeAnim *channel);   

        //Getters

        glm::mat4 getLocalTransform(); 
        std::string getBoneName(); 
        int getBoneID(); 

        //******************
        int getPosIndex(float animTime);
        int getRotIndex(float animTime);
        int getScaleIndex(float animTime);  

        void update(float animTime);
        void computeLocalTransforms(float animTime);
        fungl::Matrix4f getBoneLocalTransformMat();

    private:
        float getScaleFactor(float lastTimeStamp, float nextTimeStamp, float animTime);      
        glm::mat4 interpolatePos(float animTime);
        glm::mat4 interpolateRot(float animTime);
        glm::mat4 interpolateScale(float animTime);
        aiVector3D computeInterpolatedPosition(float animTime);
        aiQuaternion computeInterpolatedRotation(float animTime);
        aiVector3D computeInterpolatedScale(float animTime);   



};


#endif // _BONE_H_
