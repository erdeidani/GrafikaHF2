//=============================================================================================
// Mintaprogram: Zöld háromszög. Érvényes 2019. ösztöl.
//
// A beadott program csak ebben a fájlban lehet, a fájl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mást "beincludolni", illetve más könyvtárat használni
// - fáljmuveleteket végezni a printf-et kivéve
// - Máshonnan átvett programrésszleteket forrásmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Erdei Dániel Patrik
// Neptun : BAVX18
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

GPUProgram gpuProgram; 
unsigned int vao;

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

struct Cone :public Intersectable { //https://www.shadertoy.com/view/MtcXWr
	float cosa;
	float height;
	vec3 tip;
	vec3 mid;

	Cone(float _cosa, float _height, const vec3& _tip, const vec3& _mid, Material* _material) {
		cosa = _cosa;
		height = _height;
		tip = _tip;
		mid = _mid;
		material = _material;
	}
	Hit intersect(const Ray& ray) { //ez a fuggveny a fentebb emlitett webhelyrol lett kimasolva
		Hit hit;
		vec3 co = ray.start - tip;

		float a = dot(ray.dir, mid) * dot(ray.dir, mid) - dot(ray.dir, ray.dir) * cosa * cosa;
		float b = 2.0f * (dot(ray.dir, mid) * dot(co, mid) - dot(ray.dir, co) * cosa * cosa);
		float c = dot(co, mid) * dot(co, mid) - dot(co, co) * cosa * cosa;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;

		discr = sqrtf(discr);
		float t1 = (-b - discr) / (2.0f * a);
		float t2 = (-b + discr) / (2.0f * a);

		float t = t1;
		if (t < 0.0f || (t2 > 0.0f && t2 < t)) t = t2;
		if (t < 0.0f) return hit;
		if (t1 < t2)t = t1;

		vec3 cp = ray.start + t * ray.dir - tip;
		float h = dot(cp, mid);
		if (h < 0.0f || h > height) {
			t = t2;
			cp = ray.start + t * ray.dir - tip;
			h = dot(cp, mid);
			if (h < 0.0f || h > height) return hit;
		}

		vec3 n = normalize(cp * dot(mid, cp) / dot(cp, cp) - mid);

		hit.t = t;
		hit.position = ray.start + ray.dir * t;
		vec3 rp = normalize(hit.position - tip);
		hit.normal = normalize(cross(cross(n, rp), rp));
		hit.material = material;
		return hit;
	}
};

struct Light {
	Cone* cone;
	vec3 position;
	vec3 Le;

	Light(vec3 _position, vec3 _Le, Cone* _cone) {
		position = _position;
		cone = _cone;
		Le = _Le;
	}
	vec3 rad(float distance) {
		return Le / (powf(distance, 2));
	}
};

struct Triangle : public Intersectable {
	std::vector<vec3>vertices;
	vec3 normal = vec3(0, 0, 0);

	Triangle(const std::vector<vec3> _vertices, Material* _material) {
		vertices = _vertices;
		material = _material;

		vec3 r1r2 = vertices[1] - vertices[0];
		vec3 r3r1 = vertices[2] - vertices[0];
		normal = cross(r1r2, r3r1);
		normal = normalize(normal);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float t;

		float toph = dot((vertices[0] - ray.start), normal);
		float both = dot(ray.dir, normal);
		vec3 p = ray.start + ray.dir * ((float)toph / both);

		vec3 test1 = cross((vertices[1] - vertices[0]), (p - vertices[0]));
		vec3 test2 = cross((vertices[2] - vertices[1]), (p - vertices[1]));
		vec3 test3 = cross((vertices[0] - vertices[2]), (p - vertices[2]));

		t = (float)toph / both;
		if ((dot(test1, normal) < 0.0f) || (dot(test2, normal) < 0.0f) || (dot(test3, normal) < 0.0f)) {
			return hit;
		}
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normal;
		hit.material = material;
		return hit;
	}
};

struct Icosahedron {
	std::vector<vec3> vertices = {
		vec3(0, -0.525731, 0.850651),
		vec3(0.850651, 0, 0.525731),
		vec3(0.850651, 0, -0.525731),
		vec3(-0.850651, 0, -0.525731),
		vec3(-0.850651, 0, 0.525731),
		vec3(-0.525731, 0.850651, 0),
		vec3(0.525731, 0.850651, 0),
		vec3(0.525731, -0.850651, 0),
		vec3(-0.525731, -0.850651, 0),
		vec3(0, -0.525731, -0.850651),
		vec3(0, 0.525731, -0.850651),
		vec3(0, 0.525731, 0.850651)
	};

	const std::vector<vec3> faces = {
		vec3(2,3,7),
		vec3(2,8,3),
		vec3(4,5,6),
		vec3(5,4,9),
		vec3(7,6,12),
		vec3(6,7,11),
		vec3(10,11,3),
		vec3(11,10,4),
		vec3(8,9,10),
		vec3(9,8,1),
		vec3(12,1,2),
		vec3(1,12,5),
		vec3(7,3,11),
		vec3(2,7,12),
		vec3(4,6,11),
		vec3(6,5,12),
		vec3(3,8,10),
		vec3(8,2,1),
		vec3(4,10,9),
		vec3(5,9,1)
	};
};

class Scene {

	std::vector<Intersectable*> objects;
	std::vector<Light> lights;
	Camera camera;
	vec3 La, Lsl;
	Cone* spotLight1;
	Cone* spotLight2;
	Cone* spotLight3;
	Icosahedron* icosahedron;

public:
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		
		Lsl = vec3(1, 1, 1);
		La = vec3(0.25f, 0.25f, 0.25f);

		vec3 Le(1.0f, 1.0f, 1.0f);

		vec3 kd(0.2f, 0.2f, 0.2f), ks(2, 2, 2);
		vec3 kd2(0.3f, 0.3f, 0.3f);
		Material* material = new Material(kd, ks, 50);
		Material* material2 = new Material(kd2, ks, 50);

		drawBackground(material);
		drawCones(material2);
		drawTetraeder(material);

		Light red = Light(spotLight1->tip + (spotLight1->mid * 0.02f), vec3(10.0f, 0.0f, 0.0f), spotLight1);
		lights.push_back(red);
		Light green = Light(spotLight2->tip + (spotLight2->mid * 0.02f), vec3(0.0f, 10.0f, 0.0f), spotLight2);
		lights.push_back(green);
		Light blue = Light(spotLight3->tip + (spotLight3->mid * 0.02f), vec3(0.0f, 0.0f, 10.0f), spotLight3);
		lights.push_back(blue);

		drawIcosahedron(material);
	}

	void drawIcosahedron(Material* material) {
		icosahedron = new Icosahedron();
		for (vec3 face : icosahedron->faces) {
			std::vector<vec3> trianglePoints;
			trianglePoints.push_back(icosahedron->vertices[face.x - 1] * 0.25f);
			trianglePoints.push_back(icosahedron->vertices[face.y - 1] * 0.25f);
			trianglePoints.push_back(icosahedron->vertices[face.z - 1] * 0.25f);
			objects.push_back(new Triangle(trianglePoints, material));
		}
	}

	void drawBackground(Material* material) {
		std::vector<vec3> rightUpperVertices;
		rightUpperVertices.push_back(vec3(-0.1f, 0.6f, -0.9f) * 0.9f);
		rightUpperVertices.push_back(vec3(0.7f, -0.5f, 0.0f) * 0.9f);
		rightUpperVertices.push_back(vec3(0.7f, 0.6f, 0.0f) * 0.9f);
		objects.push_back(new Triangle(rightUpperVertices, material));

		std::vector<vec3> rightLowerVertices;
		rightLowerVertices.push_back(vec3(-0.1f, 0.6f, -0.9f) * 0.9f);
		rightLowerVertices.push_back(vec3(0.7f, -0.5f, 0.0f) * 0.9f);
		rightLowerVertices.push_back(vec3(-0.1f, -0.5f, -0.9f) * 0.9f);
		objects.push_back(new Triangle(rightLowerVertices, material));

		std::vector<vec3> leftUpperVertices;
		leftUpperVertices.push_back(vec3(-0.1f, -0.5f, -0.9f) * 0.9f);
		leftUpperVertices.push_back(vec3(-0.1f, 0.6f, -0.9f) * 0.9f);
		leftUpperVertices.push_back(vec3(-0.9f, 0.6f, 0.0f) * 0.9f);
		objects.push_back(new Triangle(leftUpperVertices, material));

		std::vector<vec3> leftLowerVertices;
		leftLowerVertices.push_back(vec3(-0.1f, -0.5f, -0.9f) * 0.9f);
		leftLowerVertices.push_back(vec3(-0.9f, -0.5f, 0.0f) * 0.9f);
		leftLowerVertices.push_back(vec3(-0.9f, 0.6f, 0.0f) * 0.9f);
		objects.push_back(new Triangle(leftLowerVertices, material));

		std::vector<vec3> ceilingFartherVertices;
		ceilingFartherVertices.push_back(vec3(-0.9f, 0.6f, 0.0f) * 0.9f);
		ceilingFartherVertices.push_back(vec3(-0.1f, 0.6f, -0.9f) * 0.9f);
		ceilingFartherVertices.push_back(vec3(0.7f, 0.6f, 0.0f) * 0.9f);
		objects.push_back(new Triangle(ceilingFartherVertices, material));

		std::vector<vec3> ceilingCloserVertices;
		ceilingCloserVertices.push_back(vec3(-0.9f, 0.6f, 0.0f) * 0.9f);
		ceilingCloserVertices.push_back(vec3(-0.1f, 0.6f, 0.9f) * 0.9f);
		ceilingCloserVertices.push_back(vec3(0.7f, 0.6f, 0.0f) * 0.9f);
		objects.push_back(new Triangle(ceilingCloserVertices, material));

		std::vector<vec3> floorLeftVertices;
		floorLeftVertices.push_back(vec3(-0.9f, -0.5f, 0.0f) * 0.9f);
		floorLeftVertices.push_back(vec3(-0.1f, -0.5f, -0.9f) * 0.9f);
		floorLeftVertices.push_back(vec3(-0.1f, -0.5f, 0.9f) * 0.9f);
		objects.push_back(new Triangle(floorLeftVertices, material));

		std::vector<vec3> floorRightVertices;
		floorRightVertices.push_back(vec3(-0.1f, -0.5f, -0.9f) * 0.9f);
		floorRightVertices.push_back(vec3(-0.1f, -0.5f, 0.9f) * 0.9f);
		floorRightVertices.push_back(vec3(0.7f, -0.5f, 0.0f) * 0.9f);
		objects.push_back(new Triangle(floorRightVertices, material));
	}

	void drawCones(Material* material) {

		spotLight1 = new Cone(0.9f, 0.15f, vec3(-0.2f, 0.4f, 0.4f), vec3(0.0f, -1.0f, 0.0f), material);
		spotLight2 = new Cone(0.9f, 0.15f, vec3(-0.4f, 0.1f, 0.5f), vec3(-0.1f, 0.6f, -0.7f), material);
		spotLight3 = new Cone(0.9f, 0.15f, vec3(0.4f, -0.1f, 0.3f), vec3(-1.0f, 0.0f, -0.5f), material);

		objects.push_back(spotLight1);
		objects.push_back(spotLight2);
		objects.push_back(spotLight3);
	}


	void drawTetraeder(Material* material) {
		std::vector<vec3> bottomVertices;
		bottomVertices.push_back(vec3(0.0f, -0.4f, 0.0f));
		bottomVertices.push_back(vec3(-0.6f, -0.4f, 0.0f));
		bottomVertices.push_back(vec3(-0.3f, -0.4f, 0.52f));

		std::vector<vec3> backSideVertices;
		backSideVertices.push_back(vec3(0.0f, -0.4f, 0.0f));
		backSideVertices.push_back(vec3(-0.6f, -0.4f, 0.0f));
		backSideVertices.push_back(vec3(-0.3f, 0.08f, 0.21f));

		std::vector<vec3> leftSideVertices;
		leftSideVertices.push_back(vec3(-0.3f, -0.4f, 0.52f));
		leftSideVertices.push_back(vec3(-0.6f, -0.4f, 0.0f));
		leftSideVertices.push_back(vec3(-0.3f, 0.08f, 0.21f));

		std::vector<vec3> rightSideVertices;
		rightSideVertices.push_back(vec3(0.0f, -0.4f, 0.0f));
		rightSideVertices.push_back(vec3(-0.3f, -0.4f, 0.52f));
		rightSideVertices.push_back(vec3(-0.3f, 0.08f, 0.21f));

		objects.push_back(new Triangle(bottomVertices, material));
		objects.push_back(new Triangle(backSideVertices, material));
		objects.push_back(new Triangle(leftSideVertices, material));
		objects.push_back(new Triangle(rightSideVertices, material));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
			//#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray, float maxLight) {
		for (Intersectable* object : objects)
			if (object->intersect(ray).t > 0 && object->intersect(ray).t < maxLight)
				return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return vec3(0, 0, 0);


		vec3 outRadiance = La * (1 + dot(hit.normal, -ray.dir));

		for (Light light : lights) {
			vec3 dir = light.position - hit.position;
			Ray shadowRay(hit.position + hit.normal * epsilon, dir);

			float cosTheta = dot(hit.normal, normalize(dir));

			if (cosTheta > 0 && !shadowIntersect(shadowRay, length(dir))) {	

				outRadiance = outRadiance + light.rad(length(dir)) * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + normalize(dir));
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light.rad(length(dir)) * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};

class FullScreenTexturedQuad {
	unsigned int vao;	
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
Scene scene;


void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	delete fullScreenTexturedQuad;
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();								// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {}
