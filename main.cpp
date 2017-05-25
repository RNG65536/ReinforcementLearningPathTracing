// TODO : correctly handle pdf conversion, pdfA -> pdfW

#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008 
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt 
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2 
#include <random>
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <cassert>
#include <mutex>
using std::cout;
using std::endl;
#define M_PI 3.14159265358979323846
std::uniform_real_distribution<double> dist;
std::mt19937_64 rng;
double rand01(unsigned short *x = nullptr)
{
    return dist(rng);
}
struct Vec {        // Usage: time ./smallpt 5000 && xv image.ppm 
    double x, y, z;                  // position, also color (r,g,b) 
    Vec(double x_ = 0, double y_ = 0, double z_ = 0){ x = x_; y = y_; z = z_; }
    Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
    Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
    Vec operator*(double b) const { return Vec(x*b, y*b, z*b); }
    Vec mult(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
    Vec& norm(){ return *this = *this * (1 / sqrt(x*x + y*y + z*z)); }
    double dot(const Vec &b) const { return x*b.x + y*b.y + z*b.z; } // cross: 
    Vec operator%(Vec&b){ return Vec(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x); }
    double avg() const { return (x + y + z) / 3.0; }
};

struct StratSample
{
    double x, y;
    int stratum_idx;
    double pdf;
    int n_strata;
};
class StratifiedSampler
{
    double _xmin, _xmax;
    double _ymin, _ymax;
    int _idx;

public:
    StratifiedSampler(int idx, double xmin, double xmax, double ymin, double ymax)
        : _xmin(xmin), _xmax(xmax), _ymin(ymin), _ymax(ymax), _idx(idx)
    {

    }
    StratSample sample()
    {
        StratSample s;
        s.x = rand01() * (_xmax - _xmin) + _xmin;
        s.y = rand01() * (_ymax - _ymin) + _ymin;
        s.stratum_idx = _idx;
        return s;
    }
};

const double alpha = 0.5; // learning rate
const double initial_q_value = 1.0;

class QFunc
{
    Vec _id;
    Vec _p;
    std::vector<StratifiedSampler> s;
    std::vector<double> q;

public:
    QFunc(const Vec& id, const Vec& p)
        : _id(id), _p(p)
    {
        int n = 0;
        for (int j = 0; j < 5; j++)
        {
            for (int i = 0; i < 5; i++)
            {
                s.push_back(StratifiedSampler(n++, i * 0.2, (i + 1) * 0.2, j * 0.2, (j + 1) * 0.2));
            }
        }
        q.resize(5 * 5);
        q.assign(q.size(), initial_q_value);
    }
    int size() { return q.size(); }
    StratifiedSampler& choose(double& pdf)
    {
        double inv_q_sum = 0;
        for (int i = 0; i < q.size(); i++)
        {
            inv_q_sum += q[i];
        }
        inv_q_sum = 1.0 / inv_q_sum;
        double r = rand01();

        double sum = 0;
        for (int i = 0; i < q.size(); i++)
        {
            sum += q[i];
            if (r < sum * inv_q_sum)
            {
                pdf = q[i] * inv_q_sum;
                return s[i];
            }
        }
        pdf = q[q.size() - 1] * inv_q_sum;
        return s[q.size() - 1];
    }

    double q_mean()
    {
        double ret = 0;
        for (int i = 0; i < q.size(); i++)
        {
            ret += q[i];
        }
        return ret / q.size();
    }

    void update(double value, int str_idx)
    {
        q[str_idx] = q[str_idx] * (1 - alpha) + value * alpha;
    }

//     Vec& id()
//     {
//         return _id;
//     }
//     Vec& p()
//     {
//         return _p;
//     }
};

typedef int HashId;
class HashGrid
{
    std::mutex _lock;
    std::unordered_map<HashId, QFunc*> _map;
public:
    QFunc *hash(const Vec& p)
    {
        HashId id = 0;
        double px = ceil(p.x / 5.0);
        double py = ceil(p.y / 5.0);
        double pz = ceil(p.z / 5.0);
        int x = (int)px;
        int y = (int)py;
        int z = (int)pz;
        id = x + (y << 10) + (z << 20);
        if (_map.find(id) == _map.end())
        {
            _lock.lock();
            _map[id] = new QFunc(Vec(x, y, z), Vec(px, py, pz) * 5.0);
            _lock.unlock();
        }
        auto pr = _map[id];
        assert(pr);
        return pr;
    }
    ~HashGrid()
    {
        for (auto& h : _map)
        {
            delete h.second;
        }
    }
};

std::string str(const Vec& v)
{
    std::stringstream ss;
    ss << "( " << v.x << ", " << v.y << ", " << v.z << " )";
    return ss.str();
}
struct Ray { Vec o, d; Ray(Vec o_, Vec d_) : o(o_), d(d_) {} };
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance() 
class Material
{
    Refl_t type;
    Vec color;

public:
    Material(Refl_t t, const Vec& c)
        : type(t), color(c)
    {

    }
    Vec sample(StratSample& s, const Vec& wi, const Vec& nl, Vec& weight)
    {
        switch (type)
        {
        default:
        case DIFF:
        {
            double r1 = 2 * M_PI * s.x, r2 = M_PI * s.y * 0.5;
            double cos_theta = cos(r2), sin_theta = sqrt(1 - cos_theta * cos_theta);
            Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w%u;
            Vec d = (u*cos(r1)*sin_theta + v*sin(r1)*sin_theta + w*cos_theta).norm();

            // full
//             double pdf = s.pdf * s.n_strata / (2 * M_PI * 0.5 * M_PI * sin_theta);
//             weight = color * (cos_theta / M_PI / pdf);
            //simplified
            weight = color * (cos_theta * (M_PI * sin_theta) / (s.pdf * s.n_strata));

            return d;
        }
            break;
        case SPEC:
        {
            weight = color;
            return nl * 2 * nl.dot(wi) - wi;
        }
            break;
        case REFR:
        {
            Vec refl = nl * 2 * nl.dot(wi) - wi;     // Ideal dielectric REFRACTION 
            bool into = wi.dot(nl) > 0;                // Ray from outside going in? 
            Vec orient_n = into ? nl : nl * -1;
            double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = -wi.dot(orient_n), cos2t;
            if ((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0)    // Total internal reflection 
            {
                weight = color;
                return refl;
            }
            Vec tdir = (wi*-nnt - nl*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).norm();
            double a = nt - nc, b = nt + nc, R0 = a*a / (b*b), c = 1 - (into ? -ddn : tdir.dot(nl));
            double Re = R0 + (1 - R0)*c*c*c*c*c, Tr = 1 - Re, P = .25 + .5*Re, RP = Re / P, TP = Tr / (1 - P);
            if (s.x < P) // Russian roulette 
            {
                weight = color * RP;
                return refl;
            }
            else
            {
                weight = color * TP;
                return tdir;
            }
        }
            break;
        }
    }
};
class AABB
{
    Vec _min, _max;
public:
    AABB()
    {
        _min = Vec(1e10, 1e10, 1e10);
        _max = Vec(-1e10, -1e10, -1e10);
    }

    void enclose(const Vec& p)
    {
        _min.x = std::min(_min.x, p.x);
        _min.y = std::min(_min.y, p.y);
        _min.z = std::min(_min.z, p.z);
        _max.x = std::max(_max.x, p.x);
        _max.y = std::max(_max.y, p.y);
        _max.z = std::max(_max.z, p.z);
    }
    Vec& min()
    {
        return _min;
    }
    Vec& max()
    {
        return _max;
    }
};
struct Sphere {
    double rad;       // radius 
    Vec p, e, c;      // position, emission, color 
    Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive) 
    Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
        rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
    double intersect(const Ray &r) const { // returns distance, 0 if nohit 
        Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
        double t, eps = 1e-4, b = op.dot(r.d), det = b*b - op.dot(op) + rad*rad;
        if (det<0) return 0; else det = sqrt(det);
        return (t = b - det)>eps ? t : ((t = b + det) > eps ? t : 0);
    }
    Material mat() const
    {
        return Material(refl, c);
    }
};
Sphere spheres[] = {//Scene: radius, position, emission, color, material 
    Sphere(1e5, Vec(1e5 + 1 - 2e5, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFF),//Left 
//     Sphere(1e5, Vec(-1e5 + 99 + 2e5, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF),//Rght 
    Sphere(1e5, Vec(-1e5 + 599 + 2e5, 40.8, 81.6), Vec(50, 50, 50), Vec(), DIFF),//Rght 
    Sphere(1e5, Vec(50, 40.8, 1e5 - 2e5), Vec(), Vec(.75, .75, .75), DIFF),//Back 
    Sphere(1e5, Vec(50, 40.8, -1e5 + 170 + 2e5), Vec(), Vec(), DIFF),//Frnt 
    Sphere(1e5, Vec(50, 1e5 - 2e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF),//Botm 
    Sphere(1e5, Vec(50, -1e5 + 81.6 + 2e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF),//Top 
    Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1)*.999, REFR),//Mirr 
    Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1)*.999, SPEC),//Glas 
//     Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(), DIFF) //Lite 
};
inline double clamp(double x){ return x < 0 ? 0 : x>1 ? 1 : x; }
inline int toInt(double x){ return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }
inline bool intersect(const Ray &r, double &t, int &id){
    double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
    for (int i = int(n); i--;) if ((d = spheres[i].intersect(r)) && d < t){ t = d; id = i; }
    return t<inf;
}

AABB scene_box;
HashGrid QGrid;

Vec radianceRL(const Ray &r, int depth, unsigned short *Xi,
    int last_stratum_idx = -1, const Sphere *last_obj = nullptr)
{
    double t;                               // distance to intersection 
    int id = 0;                               // id of intersected object 
    if (!intersect(r, t, id)) return Vec(); // if miss, return black 
    const Sphere &obj = spheres[id];        // the hit object 

    Vec x = r.o + r.d*t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n*-1, f = obj.c;
    scene_box.enclose(x);

    QFunc *q_curr = QGrid.hash(x);
    if (depth > 0)
    {
        QFunc *q_prev = QGrid.hash(r.o);
        double val = 0;
        if (obj.e.avg() > 0)
        {
            val = obj.e.avg();
        }
        else
        {
            val = last_obj->c.avg() * q_curr->q_mean();
        }
        q_prev->update(val, last_stratum_idx);
    }

    if (++depth > 10 || obj.e.avg() > 0) return obj.e;

    double pdf;
    auto& q1 = q_curr->choose(pdf);
    auto s = q1.sample();
    s.pdf = pdf;
    s.n_strata = q_curr->size();

    Vec weight;
    Vec new_dir = obj.mat().sample(s, r.d * -1, n, weight);
    return obj.e + f.mult(radianceRL(Ray(x, new_dir), depth, Xi, s.stratum_idx, &obj)).mult(weight);
}
// Vec radiance(const Ray &r, int depth, unsigned short *Xi,
//     int last_stratum_idx = -1, const Sphere *last_obj = nullptr)
// {
//     double t;                               // distance to intersection 
//     int id = 0;                               // id of intersected object 
//     if (!intersect(r, t, id)) return Vec(); // if miss, return black 
//     const Sphere &obj = spheres[id];        // the hit object 
// 
//     Vec x = r.o + r.d*t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n*-1, f = obj.c;
//     scene_box.enclose(x);
// 
//     if (++depth > 10 || obj.e.avg() > 0) return obj.e;
// 
// 
//     Vec weight;
//     Vec new_dir = obj.mat().sample(rand01(), rand01(), r.d * -1, n, weight);
//     return obj.e + f.mult(radiance(Ray(x, new_dir), depth, Xi)).mult(weight);
// }
int main(int argc, char *argv[]){
    int w = 256, h = 256, samps = 100; // # samples 
    Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir 
    Vec cx = Vec(w*.5135 / h), cy = (cx%cam.d).norm()*.5135, r, *c = new Vec[w*h];
#pragma omp parallel for schedule(dynamic, 1) private(r)       // OpenMP 
    for (int y = 0; y < h; y++){                       // Loop over image rows 
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100.*y / (h - 1));
        for (unsigned short x = 0, Xi[3] = { 0, 0, y*y*y }; x < w; x++)   // Loop cols 
            for (int sy = 0, i = (h - y - 1)*w + x; sy < 2; sy++)     // 2x2 subpixel rows 
                for (int sx = 0; sx < 2; sx++, r = Vec()){        // 2x2 subpixel cols 
                    for (int s = 0; s < samps; s++){
                        double r1 = 2 * rand01(Xi), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        double r2 = 2 * rand01(Xi), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                        Vec d = cx*(((sx + .5 + dx) / 2 + x) / w - .5) +
                            cy*(((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
                        r = r + radianceRL(Ray(cam.o + d * 140, d.norm()), 0, Xi)*(1. / samps);
                    } // Camera rays are pushed ^^^^^ forward to start in interior 
                    c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
                }
    }
    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file. 
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w*h; i++)
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}