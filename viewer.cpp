#include <igl/opengl/glfw/Viewer.h>
#include <igl/matrix_to_list.h>
#include <igl/serialize.h>
#include <igl/vertex_components.h>
#include <igl/facet_components.h>
#include "matchmaker.h"
#include "target_polygon.h"
#include "progressive_embedding.h"
#include "plot.h"
#include "loader.h"
#include "argh.h"
#include "slim/slim.h"
#include <map>
#include "validity_check.h"
#include <igl/remove_unreferenced.h>
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/matrix_to_list.h>
#include <igl/read_triangle_mesh.h>
#include <igl/serialize.h>
#include <igl/writeDMAT.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <igl/write_triangle_mesh.h>
#include <igl/facet_components.h>
#include <igl/remove_unreferenced.h>
#include <Eigen/Cholesky>

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include "projected_newton.hpp"
#include "opt_prepare.h"

long global_autodiff_time = 0;
long global_project_time = 0;



int main(int argc, char *argv[])
{
    std::string model = argv[1];
    Eigen::MatrixXd V, Vnew;
    Eigen::MatrixXd uv, CN;
    Eigen::MatrixXi F, Fuv, FN;

    igl::readOBJ(model, V, uv, CN, F, Fuv, FN);

    Eigen::VectorXi uv2V(uv.rows());
    for (int i = 0; i < F.rows(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            uv2V(Fuv(i, j)) = F(i, j);
        }
    }
    Vnew.resize(uv.rows(), 3);
    for (int i = 0; i < Vnew.rows(); i++)
    {
        Vnew.row(i) = V.row(uv2V(i));
    }
    V = Vnew;
    F = Fuv;

    // double ratio = (uv.row(F(0, 1)) - uv.row(F(0, 0))).norm() / (V.row(F(0, 1)) - V.row(F(0, 0))).norm();
    // uv = uv / ratio;
    std::cout << (uv.row(F(1, 1)) - uv.row(F(1, 0))).norm() << "\t V " << (V.row(F(1, 1)) - V.row(F(1, 0))).norm() << std::endl;
    
   
    Xi cut;
    std::vector<std::vector<int>> bds;
    igl::boundary_loop(F, bds);
    std::vector<int> bd = bds[0];
    std::vector<bool> is_visited(uv.rows(), false);
    for (int i = 0; i < bd.size(); i++)
    {
        int v0 = bd[i], v1 = bd[(i + 1) % bd.size()];
        if (is_visited[v0])
            continue;
        is_visited[v0] = true;
        for (int j = 0; j < bd.size(); j++)
        {
            int v2 = bd[j], v3 = bd[(j + 1) % (int)bd.size()];
            if (uv2V(v0) == uv2V(v3) && uv2V(v1) == uv2V(v2))
            {
                cut.conservativeResize(cut.rows() + 1, 4);
                cut.row(cut.rows() - 1) << v0, v1, v2, v3;
                is_visited[v2] = true;
            }
        }
    }

    // check cut
    std::cout << "check cut" << std::endl;
    for (int i = 0; i < cut.rows(); i++)
    {
        double l1 = (uv.row(cut(i, 0)) - uv.row(cut(i, 1))).norm();
        double l2 = (uv.row(cut(i, 2)) - uv.row(cut(i, 3))).norm();

        // if (fabs(l1 - l2) > 1e-5)
        if (true)
        {
            std::cout << cut.row(i) << "\tlendiff: " << fabs(l1 - l2) << std::endl;
        }
    }

    spXd Aeq;
    buildAeq(cut, uv, F, Aeq);
    spXd AeqT = Aeq.transpose();
    
//    uv(2,0) += 0.2;
    // uv(2,1) += 0.3;
    
    Vd dblarea;
    igl::doublearea(V, F, dblarea);
    dblarea *= 0.5;
    double mesh_area = dblarea.sum();

    spXd Dx, Dy, G;
    prepare(V, F, Dx, Dy);
    G = combine_Dx_Dy(Dx, Dy);

    auto compute_energy = [&G, &dblarea](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        return compute_energy_from_jacobian(Ji, dblarea);
    };

    double energy = compute_energy(uv);
    std::cout << "Start Energy" << energy << std::endl;

    auto do_opt = [&F, &Aeq, &AeqT, &G, &dblarea, &compute_energy](Eigen::MatrixXd &cur_uv, int N) {
        double energy = compute_energy(cur_uv);
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        for (int ii = 0; ii < N; ii++)
        {
            spXd hessian;
            Vd grad;
            get_grad_and_hessian(G, dblarea, cur_uv, grad, hessian);
            spXd kkt(hessian.rows() + Aeq.rows(), hessian.cols() + Aeq.rows());
            buildkkt(hessian, Aeq, AeqT, kkt);
            if (ii == 0)
                solver.analyzePattern(kkt);
            grad.conservativeResize(kkt.cols());
            for (int i = hessian.cols(); i < kkt.cols(); i++)
                grad(i) = 0;
            solver.factorize(kkt);
            std::cout << "solver.info():" << solver.info() << std::endl;
            // Xd kkt_dense(kkt);
            // std::cout << "kkt" << kkt << std::endl;
            // std::cout << "grad" << grad << std::endl;
            Vd newton = solver.solve(grad);
            newton.conservativeResize(hessian.cols());
            grad.conservativeResize(hessian.cols());
            Xd new_dir = -Eigen::Map<Xd>(newton.data(), cur_uv.rows(), 2); // newton dir
            energy = bi_linesearch(F, cur_uv, new_dir, compute_energy, grad, energy);
            std::cout << std::setprecision(20) << energy << std::endl;
        }
        return energy;
    };

    double scale = 1.0;
#ifdef VIEWER_LAUNCH
    igl::opengl::glfw::Viewer vr;
    auto key_down_new = [&](
                            igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier) {
        if (key == ' ')
        {
            energy = do_opt(uv, 1);
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            viewer.core().align_camera_center(V);
            viewer.data().set_uv(uv, F);
            viewer.data().show_texture = true;
        }
        if (key == '1')
        {
            viewer.data().clear();
            viewer.data().set_mesh(uv, F);
            viewer.core().align_camera_center(uv);
            viewer.data().show_texture = false;
        }
        if (key == ',')
        {
            scale *= 2.0;
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            viewer.core().align_camera_center(V);
            viewer.data().set_uv(uv * scale, F);
            viewer.data().show_texture = true;
        }
        if (key == '.')
        {
            scale /= 2.0;
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            viewer.core().align_camera_center(V);
            viewer.data().set_uv(uv * scale, F);
            viewer.data().show_texture = true;
        }
        return false;
    };
    vr.data().set_mesh(V, F);
    vr.callback_key_down = key_down_new;
    vr.launch();
#else
    while(std::getchar())
        do_opt(uv, 1);
#endif
}