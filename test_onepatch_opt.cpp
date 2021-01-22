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
// #include <igl/copyleft/cgal/orient2D.h>
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

void buildAeq(
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
    Aeq.resize(2 * m + 2 * bds.size(), uv.rows() * 2);

    int A, B, C, D, A2, B2, C2, D2;
    for (int i = 0; i < cut.rows(); i++)
    {
        int A2 = cut(i, 0);
        int B2 = cut(i, 1);
        int C2 = cut(i, 2);
        int D2 = cut(i, 3);

        std::complex<double> l0, l1, r0, r1;
        l0 = std::complex<double>(uv(A2, 0), uv(A2, 1));
        l1 = std::complex<double>(uv(B2, 0), uv(B2, 1));
        r0 = std::complex<double>(uv(C2, 0), uv(C2, 1));
        r1 = std::complex<double>(uv(D2, 0), uv(D2, 1));

        int r = std::round(2.0 * std::log((l0 - l1) / (r0 - r1)).imag() / igl::PI);
        r = ((r % 4) + 4) % 4; // ensure that r is between 0 and 3
        switch (r)
        {
        case 0:
            Aeq.coeffRef(c, A2) += 1;
            Aeq.coeffRef(c, B2) += -1;
            Aeq.coeffRef(c, C2) += -1;
            Aeq.coeffRef(c, D2) += 1;
            Aeq.coeffRef(c + 1, A2 + N) += 1;
            Aeq.coeffRef(c + 1, B2 + N) += -1;
            Aeq.coeffRef(c + 1, C2 + N) += -1;
            Aeq.coeffRef(c + 1, D2 + N) += 1;
            c = c + 2;
            break;
        case 1:
            Aeq.coeffRef(c, A2) += 1;
            Aeq.coeffRef(c, B2) += -1;
            Aeq.coeffRef(c, C2 + N) += 1;
            Aeq.coeffRef(c, D2 + N) += -1;
            Aeq.coeffRef(c + 1, C2) += 1;
            Aeq.coeffRef(c + 1, D2) += -1;
            Aeq.coeffRef(c + 1, A2 + N) += -1;
            Aeq.coeffRef(c + 1, B2 + N) += 1;
            c = c + 2;
            break;
        case 2:
            Aeq.coeffRef(c, A2) += 1;
            Aeq.coeffRef(c, B2) += -1;
            Aeq.coeffRef(c, C2) += 1;
            Aeq.coeffRef(c, D2) += -1;
            Aeq.coeffRef(c + 1, A2 + N) += 1;
            Aeq.coeffRef(c + 1, B2 + N) += -1;
            Aeq.coeffRef(c + 1, C2 + N) += 1;
            Aeq.coeffRef(c + 1, D2 + N) += -1;
            c = c + 2;
            break;
        case 3:
            Aeq.coeffRef(c, A2) += 1;
            Aeq.coeffRef(c, B2) += -1;
            Aeq.coeffRef(c, C2 + N) += -1;
            Aeq.coeffRef(c, D2 + N) += 1;
            Aeq.coeffRef(c + 1, C2) += 1;
            Aeq.coeffRef(c + 1, D2) += -1;
            Aeq.coeffRef(c + 1, A2 + N) += 1;
            Aeq.coeffRef(c + 1, B2 + N) += -1;
            c = c + 2;
            break;
        }
    }
    // add 2 constraints for each component
    for (auto l : bds)
    {
        std::cout << "fix " << l[0] << std::endl;
        Aeq.coeffRef(c, l[0]) = 1;
        Aeq.coeffRef(c + 1, l[0] + N) = 1;
        c = c + 2;
    }

    Aeq.makeCompressed();
    std::cout << "Aeq size " << Aeq.rows() << "," << Aeq.cols() << std::endl;
    // test initial violation
    // Eigen::VectorXd UV(uv.rows() * 2);
    // UV << uv.col(0), uv.col(1);
    // Eigen::SparseMatrix<double> t = UV.sparseView();
    // t.makeCompressed();
    // Eigen::SparseMatrix<double> mm = Aeq * t;
    // Eigen::VectorXd z = Eigen::VectorXd(mm);
    // if (z.rows() > 0)
    //     std::cout << "max violation " << z.cwiseAbs().maxCoeff() << std::endl;
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

int main(int argc, char *argv[])
{
    auto cmdl = argh::parser(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);
    if (cmdl[{"-h", "-help"}])
    {
        std::cout << "Usage: ./test_bin -options" << std::endl;
        std::cout << "-in: input model name" << std::endl;
        std::cout << "-o: output model name" << std::endl;
        std::cout << "-uv: input uv" << std::endl;
        std::cout << "-c: opt with constraints (0, 1)" << std::endl;
        exit(0);
    }

    int threshold;
    std::string model, uv_file, outfile;
    bool use_bd, space_filling_curve;
    int use_c;
    cmdl("-in") >> model;
    cmdl("-uv") >> uv_file;
    cmdl("-c") >> use_c;
    cmdl("-o", model + "_out.obj") >> outfile;

    Eigen::MatrixXd V, uv, uv_test, uv_out;
    Eigen::MatrixXi F_uv, F;
    Eigen::VectorXi S;
    std::map<std::pair<int, int>, std::vector<int>> corres;
    igl::deserialize(V, "V", model); // output the original one
    igl::deserialize(F_uv, "Fuv", model);
    igl::deserialize(F, "F_3d", model);
    igl::deserialize(uv_test, "uv", model);
    igl::deserialize(uv, "cur_uv", uv_file);
    // igl::deserialize(S, "S", model);
    igl::deserialize(corres, "corres", model);

    Xi cut;
    igl::deserialize(cut, "cut", model);
    std::cout << "cut.rows() = " << cut.rows() << std::endl;

    for (int i = 0; i < cut.rows(); i++)
    {
        double l1 = (uv.row(cut(i, 0)) - uv.row(cut(i, 1))).norm();
        double l2 = (uv.row(cut(i, 2)) - uv.row(cut(i, 3))).norm();
        double l1_test = (uv_test.row(cut(i, 0)) - uv_test.row(cut(i, 1))).norm();
        double l2_test = (uv_test.row(cut(i, 2)) - uv_test.row(cut(i, 3))).norm();
        if (fabs(l1 - l2) > 1e-5)
        {
            std::cout << cut.row(i) << "\tlendiff: " << fabs(l1 - l2);
            std::cout << "\tlendiff before: " << fabs(l1_test - l2_test) << std::endl;
        }
    }

    std::cout << V.rows() << std::endl;
    std::cout << F_uv.rows() << std::endl;
    std::cout << F.rows() << std::endl;
    std::cout << uv.rows() << std::endl;
    // std::cout << S.rows() << std::endl;
    std::cout << corres.size() << std::endl;

    std::vector<std::vector<int>> bds_uv;
    igl::boundary_loop(F_uv, bds_uv);
// compute target angle of singularities
    S.resize(uv.rows());
    S.setConstant(0);
    std::cout << "singularity angles:" << std::endl;
    for (int i : bds_uv[0])
    {
        double angle = 0;
        for (int f_id = 0; f_id < F_uv.rows(); f_id++)
        {
            for (int v_id = 0; v_id < 3; v_id++)
            {
                if (F_uv(f_id, v_id) == i)
                {
                    double l1 = (uv.row(F_uv(f_id, v_id)) - uv.row(F_uv(f_id, (v_id + 1) % 3))).norm();
                    double l2 = (uv.row(F_uv(f_id, v_id)) - uv.row(F_uv(f_id, (v_id + 2) % 3))).norm();
                    double l3 = (uv.row(F_uv(f_id, (v_id + 2) % 3)) - uv.row(F_uv(f_id, (v_id + 1) % 3))).norm();
                    double cos_a = (l1 * l1 + l2 * l2 - l3 * l3) / 2 / (l1 * l2);
                    angle += std::acos(cos_a);
                }
            }
        }
        S(i) = std::floor(angle / 2 / igl::PI);
        std::cout << i << " " << angle << " " << S(i) << std::endl;
    }

    std::cout << "bds_uv sizse = " << bds_uv.size() << std::endl;
    for (auto bd : bds_uv)
    {
        for (int i : bd)
            std::cout << i << " ";
        std::cout << std::endl;
    }

    for (auto it = corres.begin(); it != corres.end(); it++)
    {
        std::cout << "(" << it->first.first << "," << it->first.second << "):(";
        for (int v : it->second)
            std::cout << v << ",";
        std::cout << ")" << std::endl;
    }

    // igl::opengl::glfw::Viewer viewer;
    // viewer.data().set_mesh(uv, F_uv);
    // viewer.launch();
    // return 0;
    Eigen::MatrixXd c;
    Eigen::VectorXi ci;
    Eigen::MatrixXd uv_new;

    Eigen::VectorXi T, R, mark;
    Eigen::MatrixXd polygon;

    Eigen::VectorXi V_map(uv.rows());
    V_map.setConstant(-1);

    auto bd = bds_uv[0];
    for (int i = 0; i < bd.size(); i++)
    {
        int v1 = bd[i], v2 = bd[(i + 1) % bd.size()];
        std::vector<int> v_list = corres[std::pair<int, int>(v1, v2)];
        int size0 = polygon.rows();
        polygon.conservativeResize(size0 + v_list.size() - 1, 2);
        T.conservativeResize(size0 + v_list.size() - 1);
        R.conservativeResize(size0 + v_list.size() - 1);
        mark.conservativeResize(size0 + v_list.size() - 1);
        for (int j = 0; j < v_list.size() - 1; j++)
        {
            double r = (double)j / (double)(v_list.size() - 1);
            polygon.row(size0 + j) = (1 - r) * uv.row(v1) + r * uv.row(v2);
            T(size0 + j) = v_list[j];
            if (j == 0 && S(v1) != 0)
                R(size0 + j) = S(v1);
            else
                R(size0 + j) = 0;
            if (j == 0)
                mark(size0 + j) = 1;
            else
                mark(size0 + j) = 0;

            if (j == 0)
                V_map(v1) = size0 + j;
        }
    }
    std::cout << "check V_map" << std::endl;
    Xi cut_new;
    int Psize = polygon.rows();
    std::cout << "Psize = " << Psize << std::endl;
    for (int i = 0; i < cut.rows(); i++)
    {
        int l0 = (V_map(cut(i, 1)) - V_map(cut(i, 0)) + polygon.rows()) % polygon.rows();
        int l1 = (V_map(cut(i, 3)) - V_map(cut(i, 2)) + polygon.rows()) % polygon.rows();
        if (l0 != l1)
        {
            std::cout << "pair error, diff = " << l0 - l1 << std::endl;
        }
        int s0 = V_map(cut(i, 0)), e0 = V_map(cut(i, 1));
        int s1 = V_map(cut(i, 2)), e1 = V_map(cut(i, 3));
        for (int k = 0; k < l1; k++)
        {
            cut_new.conservativeResize(cut_new.rows() + 1, 6);
            cut_new.row(cut_new.rows() - 1) << (s0 + k) % Psize, (s0 + k + 1) % Psize, (s1 + l1 - k - 1) % Psize, (s1 + l1 - k) % Psize, cut(i, 4), cut(i, 5);
        }
    }
    cut = cut_new;
    std::cout << "check new cut" << std::endl;
    for (int i = 0; i < cut.rows(); i++)
    {
        double l1 = (polygon.row(cut(i, 0)) - polygon.row(cut(i, 1))).norm();
        double l2 = (polygon.row(cut(i, 2)) - polygon.row(cut(i, 3))).norm();

        if (fabs(l1 - l2) > 1e-5)
        {
            std::cout << cut.row(i) << "\tlendiff: " << fabs(l1 - l2) << std::endl;
        }
    }

    // return 0;
    std::cout << R.rows() << " " << T.rows() << " " << polygon.rows();

    match_maker(V, F, uv_new, c, ci, R, T, polygon, mark);

    auto cut_copy = cut;
    for (int i = 0; i < polygon.rows(); i++)
    {
        for (int j = 0; j < uv_new.rows(); j++)
        {
            if (polygon.row(i) == uv_new.row(j))
            {
                std::cout << "polygon(" << i << ") = uv_new(" << j << ")" << std::endl;
                for (int cut_row = 0; cut_row < cut.rows(); cut_row++)
                {
                    for (int cut_col = 0; cut_col < 4; cut_col++)
                    {
                        if (cut_copy(cut_row, cut_col) == i)
                            cut(cut_row, cut_col) = j;
                    }
                }
                break;
            }
        }
    }

    std::cout << "check new cut" << std::endl;
    for (int i = 0; i < cut.rows(); i++)
    {
        double l1 = (uv_new.row(cut(i, 0)) - uv_new.row(cut(i, 1))).norm();
        double l2 = (uv_new.row(cut(i, 2)) - uv_new.row(cut(i, 3))).norm();

        if (fabs(l1 - l2) > 1e-5)
        {
            std::cout << cut.row(i) << "\tlendiff: " << fabs(l1 - l2) << std::endl;
        }
    }

    // prepare for opt
    // Xd cur_uv = uv_new;
    // build constraints
    spXd Aeq;
    buildAeq(cut, uv_new, F, Aeq);
    spXd AeqT = Aeq.transpose();

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

    double energy = compute_energy(uv_new);
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
            for (int i = hessian.cols() + 1; i < kkt.cols(); i++)
                grad(i) = 0;
            solver.factorize(kkt);
            // if (ii == 0)
            //     solver.analyzePattern(hessian);
            // solver.factorize(hessian);

            Vd newton = solver.solve(grad);
            newton.conservativeResize(hessian.cols());
            grad.conservativeResize(hessian.cols());
            Xd new_dir = -Eigen::Map<Xd>(newton.data(), cur_uv.rows(), 2); // newton dir
            energy = bi_linesearch(F, cur_uv, new_dir, compute_energy, grad, energy);
            std::cout << std::setprecision(20) << energy << std::endl;
        }
        return energy;
    };
    // return 0;

    igl::SLIMData sData;
    sData.slim_energy = igl::SLIMData::SYMMETRIC_DIRICHLET;
    igl::SLIMData::SLIM_ENERGY energy_type = igl::SLIMData::SYMMETRIC_DIRICHLET;
    //Eigen::SparseMatrix<double> Aeq;
    Eigen::VectorXd E;
    slim_precompute(V, F, uv_new, sData, igl::SLIMData::SYMMETRIC_DIRICHLET, ci, c, 0, true, E, 1.0);
    igl::opengl::glfw::Viewer vr;
    vr.data().set_mesh(V, F);
    double scale = 1.0;
    auto key_down = [&](
                        igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier) {
        if (key == ' ')
        {
            slim_solve(sData, 20, E);
            // std::cout << E.maxCoeff() << std::endl;
            std::cout << E.sum() / E.rows() << std::endl;
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            // for (int i = 0; i < 3; i++)
            //     viewer.data().add_points(V.row(ci(i)), Eigen::RowVector3d(1, 0, 0));
            viewer.core().align_camera_center(V);
            viewer.data().set_uv(sData.V_o, F);
            viewer.data().show_texture = true;
        }
        if (key == '1')
        {
            slim_solve(sData, 20, E);
            viewer.data().clear();
            viewer.data().set_mesh(sData.V_o, F);
            // for (int i = 0; i < 3; i++)
            //     viewer.data().add_points(sData.V_o.row(ci(i)), Eigen::RowVector3d(1, 0, 0));
            viewer.core().align_camera_center(sData.V_o);
            viewer.data().show_texture = false;
        }
        if (key == ',')
        {
            scale *= 2.0;
            viewer.data().set_mesh(V, F);
            viewer.core().align_camera_center(V);
            viewer.data().set_uv(sData.V_o * scale, F);
            viewer.data().show_texture = true;
        }
        if (key == '.')
        {
            scale /= 2.0;
            viewer.data().set_mesh(V, F);
            viewer.core().align_camera_center(V);
            viewer.data().set_uv(sData.V_o * scale, F);
            viewer.data().show_texture = true;
        }
        return false;
    };
    auto key_down_new = [&](
                            igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier) {
        if (key == ' ')
        {
            energy = do_opt(uv_new, 50);
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            viewer.core().align_camera_center(V);
            viewer.data().set_uv(uv_new, F);
            viewer.data().show_texture = true;
        }
        if (key == '1')
        {
            viewer.data().clear();
            viewer.data().set_mesh(uv_new, F);
            viewer.core().align_camera_center(uv_new);
            viewer.data().show_texture = false;
        }
        if (key == ',')
        {
            scale *= 2.0;
            viewer.data().set_mesh(V, F);
            viewer.core().align_camera_center(V);
            viewer.data().set_uv(uv_new * scale, F);
            viewer.data().show_texture = true;
        }
        if (key == '.')
        {
            scale /= 2.0;
            viewer.data().set_mesh(V, F);
            viewer.core().align_camera_center(V);
            viewer.data().set_uv(uv_new * scale, F);
            viewer.data().show_texture = true;
        }
        return false;
    };
    if (use_c == 1)
        vr.callback_key_down = key_down_new;
    else
        vr.callback_key_down = key_down;
    //plot_mesh(vr,uv,F,{},Eigen::VectorXi());
    vr.launch();

    Eigen::MatrixXd CN;
    Eigen::MatrixXi FN;
    igl::writeOBJ(outfile, V, F, CN, FN, uv_new, F);
}
