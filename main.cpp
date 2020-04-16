#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		
#include <GL/freeglut.h>	
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

struct vec2 {
	float x, y;
	vec2(float _x = 0, float _y = 0) { x = _x; y = _y; }
};

struct vec3 {
	float x, y, z;
	vec3(float _x = 0, float _y = 0, float _z = 0) { x = _x; y = _y; z = _z; }
	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	vec3 operator-() const { return vec3(-x, -y, -z); }
	vec3 normalize() const { return (*this) * (1.0f / (Length() + 0.000001)); }
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	void SetUniform(unsigned shaderProg, char * name) {
		int location = glGetUniformLocation(shaderProg, name);
		if (location >= 0) glUniform3fv(location, 1, &x);
		else printf("uniform %s cannot be set\n", name);
	}
};

float dot(const vec3& v1, const vec3& v2) { return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }

vec3 cross(const vec3& v1, const vec3& v2) { return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }

struct vec4 {
	float x, y, z, w;
	vec4(float _x = 0, float _y = 0, float _z = 0, float _w = 1) { x = _x; y = _y; z = _z; w = _w; }

	void SetUniform(unsigned shaderProg, char * name) {
		int location = glGetUniformLocation(shaderProg, name);
		if (location >= 0) glUniform4fv(location, 1, &x);
		else printf("uniform %s cannot be set\n", name);
	}
};


struct mat4 { 
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}

	void SetUniform(unsigned shaderProg, char * name) {
		int location = glGetUniformLocation(shaderProg, name);   	
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, &m[0][0]);
		else printf("uniform %s cannot be set\n", name);		
	}
};

mat4 TranslateMatrix(vec3 t) {
	return mat4(1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				t.x, t.y, t.z, 1);
}

mat4 ScaleMatrix(vec3 s) {
	return mat4(s.x, 0, 0, 0,
				0, s.y, 0, 0,
				0, 0, s.z, 0,
				0, 0, 0, 1);
}

mat4 RotationMatrix(float angle, vec3 w) {
	float c = cosf(angle), s = sinf(angle);
	w = w.normalize();
	return mat4(c * (1 - w.x*w.x) + w.x*w.x, w.x*w.y*(1 - c) + w.z*s,     w.x*w.z*(1 - c) - w.y*s,     0,
		        w.x*w.y*(1 - c) - w.z*s,     c * (1 - w.y*w.y) + w.y*w.y, w.y*w.z*(1 - c) + w.x*s,     0,
				w.x*w.z*(1 - c) + w.y*s,     w.y*w.z*(1 - c) - w.x*s,     c * (1 - w.z*w.z) + w.z*w.z, 0,
				0,                           0,                           0,                           1);
}

struct Camera { 
	vec3 wEye, wLookat, wVup;   
	float fov, asp, fp, bp;		
public:
	Camera() {
		asp = 1;
		fov = 60.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 10;
	}
	mat4 V() { 
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);
		return TranslateMatrix(-wEye) * mat4(u.x, v.x, w.x, 0,
											 u.y, v.y, w.y, 0,
											 u.z, v.z, w.z, 0,
											 0,   0,   0,   1);
	}
	mat4 P() { 
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
					0, 1 / tan(fov / 2), 0, 0,
					0, 0, -(fp + bp) / (bp - fp), -1,
					0, 0, -2 * fp*bp / (bp - fp), 0);
	}
	void Animate(float t) { }
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;

	void Animate(float t) {	}
};

struct Texture {
	unsigned int textureId;

	Texture(const int width, const int height) {  glGenTextures(1, &textureId); }

	void SetUniform(unsigned shaderProg, char * samplerName, unsigned int textureUnit = 0) {
		int location = glGetUniformLocation(shaderProg, samplerName);
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		} else printf("uniform %s cannot be set\n", samplerName);
	}
};

struct CheckerBoardTexture : public Texture {
	CheckerBoardTexture(const int width = 0, const int height = 0) : Texture(width, height) {
		glBindTexture(GL_TEXTURE_2D, textureId);    
		std::vector<vec3> image(width * height);
		const vec3 yellow(1, 1, 0), blue(0, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

struct RenderState {
	mat4	  MVP, M, Minv, V, P;
	Material* material;
	Light     light;
	Texture*  texture;
	vec3	  wEye;
};

class Shader {
	void getErrorInfo(unsigned int handle) {
		int logLen, written;
		glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0) {
			char * log = new char[logLen];
			glGetShaderInfoLog(handle, logLen, &written, log);
			printf("Shader log:\n%s", log);
			delete log;
		}
	}
	void checkShader(unsigned int shader, char * message) { 	
		int OK;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
		if (!OK) { printf("%s!\n", message); getErrorInfo(shader); getchar(); }
	}
	void checkLinking(unsigned int program) { 	
		int OK;
		glGetProgramiv(program, GL_LINK_STATUS, &OK);
		if (!OK) { printf("Failed to link shader program!\n"); getErrorInfo(program); getchar(); }
	}
protected:
	unsigned int shaderProgram;
public:
	void Create(const char * vertexSource, const char * fragmentSource, const char * fsOuputName) {
		
		unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
		if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
		glShaderSource(vertexShader, 1, &vertexSource, NULL);
		glCompileShader(vertexShader);
		checkShader(vertexShader, "Vertex shader error");

		
		unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
		glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
		glCompileShader(fragmentShader);
		checkShader(fragmentShader, "Fragment shader error");

		
		shaderProgram = glCreateProgram();
		if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);

		
		glBindFragDataLocation(shaderProgram, 0, fsOuputName);	

		
		glLinkProgram(shaderProgram);
		checkLinking(shaderProgram);
	}
	virtual void Bind(RenderState state) = 0;
	~Shader() { glDeleteProgram(shaderProgram); }
};

class GouraudShader : public Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; 
		uniform vec4  wLiPos;       
		uniform vec3  wEye;         
		uniform vec3  kd, ks, ka; 
		uniform vec3  La, Le;     
		uniform float shine;     

		layout(location = 0) in vec3  vtxPos;            
		layout(location = 1) in vec3  vtxNorm;      	 

		out vec3 radiance;		    

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; 
			
			vec4 wPos = vec4(vtxPos, 1) * M;	
			vec3 L = normalize(wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w);
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			vec3 H = normalize(L + V);
			float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
			radiance = ka * La + (kd * cost + ks * pow(cosd,shine)) * Le;
		}
	)";

	
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      
		out vec4 fragmentColor; 

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(shaderProgram); 		
		state.MVP.SetUniform(shaderProgram, "MVP");
		state.M.SetUniform(shaderProgram, "M");
		state.Minv.SetUniform(shaderProgram, "Minv");
		state.wEye.SetUniform(shaderProgram, "wEye");
		state.material->kd.SetUniform(shaderProgram, "kd");
		state.material->ks.SetUniform(shaderProgram, "ks");
		state.material->ka.SetUniform(shaderProgram, "ka");
		int location = glGetUniformLocation(shaderProgram, "shine");
		if (location >= 0) glUniform1f(location, state.material->shininess); else printf("uniform shininess cannot be set\n");
		state.light.La.SetUniform(shaderProgram, "La");
		state.light.Le.SetUniform(shaderProgram, "Le");
		state.light.wLightPos.SetUniform(shaderProgram, "wLiPos");
	}
};

class PhongShader : public Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; 
		uniform vec4  wLiPos;       
		uniform vec3  wEye;         

		layout(location = 0) in vec3  vtxPos;            
		layout(location = 1) in vec3  vtxNorm;      	 
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    
		out vec3 wView;             
		out vec3 wLight;		    
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; 
		   
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;

		   texcoord = vtxUV;
		}
	)";

	
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		uniform vec3 kd, ks, ka; 
		uniform vec3 La, Le;     
		uniform float shine;     
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       
		in  vec3 wView;         
		in  vec3 wLight;        
		in vec2 texcoord;
		out vec4 fragmentColor; 

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			vec3 L = normalize(wLight);
			vec3 H = normalize(L + V);
			float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;

			
			vec3 color = ka * texColor * La + (kd * texColor * cost + ks * pow(cosd,shine)) * Le;
			fragmentColor = vec4(color, 1);
		}
	)";
public:
	PhongShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(shaderProgram); 		
		state.MVP.SetUniform(shaderProgram, "MVP");
		state.M.SetUniform(shaderProgram, "M");
		state.Minv.SetUniform(shaderProgram, "Minv");
		state.wEye.SetUniform(shaderProgram, "wEye");
		state.material->kd.SetUniform(shaderProgram, "kd");
		state.material->ks.SetUniform(shaderProgram, "ks");
		state.material->ka.SetUniform(shaderProgram, "ka");
		int location = glGetUniformLocation(shaderProgram, "shine");
		if (location >= 0) glUniform1f(location, state.material->shininess); else printf("uniform shininess cannot be set\n");
		state.light.La.SetUniform(shaderProgram, "La");
		state.light.Le.SetUniform(shaderProgram, "Le");
		state.light.wLightPos.SetUniform(shaderProgram, "wLiPos");
		state.texture->SetUniform(shaderProgram, "diffuseTexture"); 
	}
};

class NPRShader : public Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; 
		uniform vec4  wLiPos;       
		uniform vec3  wEye;         

		layout(location = 0) in vec3  vtxPos;            
		layout(location = 1) in vec3  vtxNorm;      	 
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; 
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	
		in  vec2 texcoord;
		out vec4 fragmentColor;    			

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(shaderProgram); 		
		state.MVP.SetUniform(shaderProgram, "MVP");
		state.M.SetUniform(shaderProgram, "M");
		state.Minv.SetUniform(shaderProgram, "Minv");
		state.wEye.SetUniform(shaderProgram, "wEye");
		state.light.wLightPos.SetUniform(shaderProgram, "wLiPos");
		state.texture->SetUniform(shaderProgram, "diffuseTexture");
	}
};

struct VertexData {
	vec3 position, normal;
	vec2 texcoord;
};

class Geometry {
	unsigned int vao, type;        
protected: 
	int nVertices;
public:
	Geometry(unsigned int _type) {
		type = _type;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(type, 0, nVertices);
	}
};

class ParamSurface : public Geometry {
public:
	ParamSurface() : Geometry(GL_TRIANGLES) {}

	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = 16, int M = 16) {
		unsigned int vbo;
		glGenBuffers(1, &vbo); 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVertices = N * M * 6;
		std::vector<VertexData> vtxData;	
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				vtxData.push_back(GenVertexData((float)i / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)(j + 1) / M));
				vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVertices * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		
		glEnableVertexAttribArray(0);  
		glEnableVertexAttribArray(1);  
		glEnableVertexAttribArray(2);  
		
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position)); 
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
};

class Sphere : public ParamSurface {
public:
	Sphere() { Create(20, 20); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position = vd.normal = vec3(cosf(u * 2.0f * M_PI) * sinf(v*M_PI), sinf(u * 2.0f * M_PI) * sinf(v*M_PI), cosf(v*M_PI));
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class Torus : public ParamSurface {
	const float R = 1, r = 0.5;

	vec3 Point(float u, float v, float rr) {
		float ur = u * 2.0f * M_PI, vr = v * 2.0f * M_PI;
		float l = R + rr * cosf(ur);
		return vec3(l * cosf(vr), l * sinf(vr), rr * sinf(ur));
	}
public:
	Torus() { Create(40, 40); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position = Point(u, v, r);
		vd.normal = (vd.position - Point(u, v, 0)) * (1.0f / r);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

struct Object {
	Shader * shader;
	Material * material;
	Texture * texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	void Animate(float tstart, float tend) { rotationAngle = 0.8 * tend; }
};

class Scene {
	std::vector<Object *> objects;
public:
	Camera camera; 
	Light light;

	void Build() {
		
		Shader * phongShader = new PhongShader();
		Shader * gouraudShader = new GouraudShader();
		Shader * nprShader = new NPRShader();

		
		Material * material0 = new Material;
		material0->kd = vec3(1.0f, 0.1f, 0.2f);
		material0->ks = vec3(1, 1, 1);
		material0->ka = vec3(0.2f, 0.2f, 0.2f);
		material0->shininess = 50;

		Material * material1 = new Material;
		material1->kd = vec3(0, 1, 1);
		material1->ks = vec3(2, 2, 2);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 200;

		
		Texture * texture4x8 = new CheckerBoardTexture(4, 8);
		Texture * texture15x20 = new CheckerBoardTexture(15, 20);

		
		Geometry * torus = new Torus();
		Geometry * sphere = new Sphere();

		
		Object * torusObject1 = new Object(phongShader, material0, texture4x8, torus);
		torusObject1->translation = vec3(-1, -1, 0);
		torusObject1->rotationAxis = vec3(1, 1, 1);
		torusObject1->scale = vec3(0.7f, 0.7f, 0.7f);
		objects.push_back(torusObject1);

		Object * sphereObject1 = new Object(phongShader, material1, texture15x20, sphere);
		sphereObject1->translation = vec3(1, -1, 0);
		sphereObject1->rotationAxis = vec3(0, 1, 1);
		sphereObject1->scale = vec3(0.5f, 1.2f, 0.5f);
		objects.push_back(sphereObject1);

		Object * torusObject2 = new Object(nprShader, NULL, texture4x8, torus);
		torusObject2->translation = vec3(-1, 1, -1);
		torusObject2->rotationAxis = vec3(1, 1, -1);
		torusObject2->scale = vec3(0.7f, 0.7f, 0.7f);
		objects.push_back(torusObject2);

		Object * sphereObject2 = new Object(gouraudShader, material1, NULL, sphere);
		sphereObject2->translation = vec3(1, 1, -1);
		sphereObject2->rotationAxis = vec3(0, 1, -1);
		sphereObject2->scale = vec3(0.5f, 1.2f, 0.5f);
		objects.push_back(sphereObject2);

		
		camera.wEye = vec3(0, 0, 4);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		
		light.wLightPos = vec4(5, 5, 4, 0);	
		light.La = vec3(1, 1, 1);
		light.Le = vec3(3, 3, 3);
	}
	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.light = light;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		camera.Animate(tend);
		light.Animate(tend);
		for (Object * obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;
 
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	scene.Render();
	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) { }

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	static float tend = 0;
	const float dt = 0.1; 
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt); 
	}
	glutPostRedisplay();
}

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(3, 3);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				
	glutInitWindowPosition(100, 100);							
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	
	glewInit();
#endif
	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	int majorVersion, minorVersion;
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	return 1;
}

