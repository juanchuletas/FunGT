#version 440


layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 normal_position;
layout (location = 2) in vec2 texture_position;
layout (location = 3) in ivec4 bone_ids; 
layout (location = 4) in vec4 bone_weights;

//Here come the uniform value form the matrices
const int maxBonesAllowed = 300; 
const int maxBoneInfluencePerVertex = 4; 
//uniforms that main program sends
uniform mat4 ModelMatrix; 
uniform mat4 ViewMatrix; 
uniform mat4 ProjectionMatrix; 
uniform mat4 finalBonesMatrix[maxBonesAllowed];



//Outputs ti the fragment shader: 
out vec2 textureCoords;
out vec3 FragPos; 
out vec3 Normal; 

void main(){

    vec4 totalPos = vec4(0.0f);
    vec3 totalNorm = vec3(0.0f);
    for(int i = 0; i<maxBoneInfluencePerVertex; i++){
  
        if(bone_weights[i]>0.0){
            vec4 localPos = finalBonesMatrix[bone_ids[i]]*vec4(vertex_position,1.0f);
            totalPos += localPos*bone_weights[i];
            vec3 localNormal = mat3(finalBonesMatrix[bone_ids[i]])*normal_position;
            totalNorm += localNormal; 
        }   
    }
    FragPos = vec3(totalPos); 
    Normal  =  mat3(transpose(inverse(ModelMatrix)))*totalNorm; 
    gl_Position = ProjectionMatrix*ViewMatrix*ModelMatrix*totalPos;

    //Sends the textures to the fragment shader: 
    textureCoords =  texture_position;


}