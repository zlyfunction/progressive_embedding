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
#include <igl/boundary_loop.h>
#include <igl/cat.h>
#include <igl/doublearea.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/grad.h>
#include <igl/harmonic.h>
#include <igl/local_basis.h>
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
#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include "projected_newton.hpp"

long global_autodiff_time = 0;
long global_project_time = 0;

double buildAeq(
    const Eigen::MatrixXi &cut,
    const Eigen::MatrixXd &uv,
    const Eigen::MatrixXi &F,
    Eigen::SparseMatrix<double> &Aeq)
{
    std::cout << "build constraint matrix\n";
    Eigen::VectorXd tail;
    int N = uv.rows();
    int c = 0;
    int m = cut.rows();

    std::vector<std::vector<int>> bds;
    igl::boundary_loop(F, bds);

    std::cout << "#components = " << bds.size() << std::endl;
    // Aeq.resize(2 * m, uv.rows() * 2);
    // try to fix 2 dof for each component
    // Aeq.resize(2 * m + 2 * bds.size(), uv.rows() * 2);

    // for harmonic
    
#define CONSTRAINTS
#ifdef CONSTRAINTS
    Aeq.resize(2 * m + 3 * bds.size(), uv.rows() * 2);
    int A, B, C, D, A2, B2, C2, D2;
    for (int i = 0; i < cut.rows(); i++)
    {
        int A2 = cut(i, 0);
        int B2 = cut(i, 1);
        int C2 = cut(i, 2);
        int D2 = cut(i, 3);

        Eigen::Vector2d e_ab = uv.row(B2) - uv.row(A2);
        Eigen::Vector2d e_dc = uv.row(C2) - uv.row(D2);

        Eigen::Vector2d e_ab_perp;
        e_ab_perp(0) = -e_ab(1);
        e_ab_perp(1) = e_ab(0);
        double angle = std::atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));

        int r = (int)(std::round(2 * angle / igl::PI) + 2) % 4;

        std::vector<Eigen::Matrix2d> r_mat(4);
        r_mat[0] << -1, 0, 0, -1;
        r_mat[1] << 0, 1, -1, 0;
        r_mat[2] << 1, 0, 0, 1;
        r_mat[3] << 0, -1, 1, 0;

        Aeq.coeffRef(c, A2) += 1;
        Aeq.coeffRef(c, B2) += -1;
        Aeq.coeffRef(c + 1, A2 + N) += 1;
        Aeq.coeffRef(c + 1, B2 + N) += -1;

        Aeq.coeffRef(c, C2) += r_mat[r](0, 0);
        Aeq.coeffRef(c, D2) += -r_mat[r](0, 0);
        Aeq.coeffRef(c, C2 + N) += r_mat[r](0, 1);
        Aeq.coeffRef(c, D2 + N) += -r_mat[r](0, 1);
        Aeq.coeffRef(c + 1, C2) += r_mat[r](1, 0);
        Aeq.coeffRef(c + 1, D2) += -r_mat[r](1, 0);
        Aeq.coeffRef(c + 1, C2 + N) += r_mat[r](1, 1);
        Aeq.coeffRef(c + 1, D2 + N) += -r_mat[r](1, 1);
        c = c + 2;
    }

    auto Aeq_no_fix = Aeq;
    Vd flat_uv = Eigen::Map<const Vd>(uv.data(), uv.size());
    Aeq_no_fix.makeCompressed();
    auto res = Aeq_no_fix * flat_uv;
    std::cout << "check constraints:" << res.cwiseAbs().maxCoeff() << std::endl;
#else
    Aeq.resize(3 * bds.size(), uv.rows() * 2);
#endif
    // add 3 constraints for each component
    for (auto l : bds)
    {
        std::cout << "fix " << l[0] << std::endl;
        Aeq.coeffRef(c, l[0]) = 1;
        Aeq.coeffRef(c + 1, l[0] + N) = 1;
        c = c + 2;

        // for harmonic
        std::cout << "fix " << l[1] << std::endl;
        Aeq.coeffRef(c, l[1]) = 1;
        // Aeq.coeffRef(c + 1, l[1] + N) = 1;
        c = c + 1;
    }
    
    std::cout << "bd loop:" << std::endl;
    for (auto v : bds[0]) std::cout << v << " ";
    std::cout << std::endl;
    Aeq.makeCompressed();
    std::cout << "Aeq size " << Aeq.rows() << "," << Aeq.cols() << std::endl;
    return 0;
    // return res.cwiseAbs().maxCoeff();
}

void buildkkt(spXd &hessian, spXd &Aeq, spXd &AeqT, spXd &kkt)
{
    // std::cout << "build kkt\n";
    kkt.reserve(hessian.nonZeros() + Aeq.nonZeros() + AeqT.nonZeros());
    for (Eigen::Index c = 0; c < kkt.cols(); ++c)
    {
        kkt.startVec(c);
        if (c < hessian.cols())
        {
            for (Eigen::SparseMatrix<double>::InnerIterator ithessian(hessian, c); ithessian; ++ithessian)
                kkt.insertBack(ithessian.row(), c) = ithessian.value();
            for (Eigen::SparseMatrix<double>::InnerIterator itAeq(Aeq, c); itAeq; ++itAeq)
                kkt.insertBack(itAeq.row() + hessian.rows(), c) = itAeq.value();
        }
        else
        {
            for (Eigen::SparseMatrix<double>::InnerIterator itAeqT(AeqT, c - hessian.cols()); itAeqT; ++itAeqT)
                kkt.insertBack(itAeqT.row(), c) = itAeqT.value();
        }
    }
    kkt.finalize();
}

void prepare(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, spXd &Dx,
             spXd &Dy)
{
    Eigen::MatrixXd F1, F2, F3;
    igl::local_basis(V, F, F1, F2, F3);
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);
    auto face_proj = [](Eigen::MatrixXd &F) {
        std::vector<Eigen::Triplet<double>> IJV;
        int f_num = F.rows();
        for (int i = 0; i < F.rows(); i++)
        {
            IJV.push_back(Eigen::Triplet<double>(i, i, F(i, 0)));
            IJV.push_back(Eigen::Triplet<double>(i, i + f_num, F(i, 1)));
            IJV.push_back(Eigen::Triplet<double>(i, i + 2 * f_num, F(i, 2)));
        }
        Eigen::SparseMatrix<double> P(f_num, 3 * f_num);
        P.setFromTriplets(IJV.begin(), IJV.end());
        return P;
    };

    Dx = face_proj(F1) * G;
    Dy = face_proj(F2) * G;
}

spXd combine_Dx_Dy(const spXd &Dx, const spXd &Dy)
{
    // [Dx, 0; Dy, 0; 0, Dx; 0, Dy]
    spXd hstack = igl::cat(1, Dx, Dy);
    spXd empty(hstack.rows(), hstack.cols());
    // gruesom way for Kronecker product.
    return igl::cat(1, igl::cat(2, hstack, empty), igl::cat(2, empty, hstack));
}

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
    auto bd = bds[0];
    std::vector<bool> is_visited(bd.size(), false);
    for (int i = 0; i < bd.size(); i++)
    {
        int v0 = bd[i], v1 = bd[(i + 1) % bd.size()];
        if (is_visited[v0])
            continue;
        is_visited[v0] = true;
        for (int j = 0; j < bd.size(); j++)
        {
            int v2 = bd[j], v3 = bd[(j + 1) % bd.size()];
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
    uv(2,1) += 0.3;
    
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
            Xd kkt_dense(kkt);
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
    // char c;
    // while(std::getchar())
    //     do_opt(uv, 1);
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
}