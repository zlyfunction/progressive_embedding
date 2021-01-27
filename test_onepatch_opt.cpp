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
#include <set>
#include "opt_prepare.h"
#include <time.h>
long global_autodiff_time = 0;
long global_project_time = 0;

void check_total_angle(const Xd &uv, const Xi &F, const Xi &cut, const Vd &Sn, const std::string &name)
{
    Vd angles(uv.rows());
    angles.setConstant(0);
    std::vector<std::vector<int>> bds;
    igl::boundary_loop(F, bds);
    auto bd = bds[0];
    for (int i = 0; i < F.rows(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int v1 = F(i, j), v2 = F(i, (j+1) % 3), v0 = F(i, (j+2) % 3);
            Eigen::Vector2d e10 = uv.row(v0) - uv.row(v1);
            Eigen::Vector2d e12 = uv.row(v2) - uv.row(v1);
            Eigen::Vector2d e10_perp;
            e10_perp(0) = -e10(1); 
            e10_perp(1) = e10(0);
            angles(v1) += std::atan2(-e10_perp.dot(e12), e10.dot(e12));
        }
    }
    

    std::vector<std::set<int>> VV(uv.rows());

    for (int i = 0; i < VV.size(); i++) VV[i].insert(i);
    for (int i = 0; i < cut.rows(); i++)
    {
        for (int v : VV[cut(i, 3)])
            VV[cut(i,0)].insert(VV[v].begin(), VV[v].end());       
        for (int v : VV[cut(i, 0)])
            VV[v] = VV[cut(i, 0)];
        for (int v : VV[cut(i, 2)])
            VV[cut(i,1)].insert(VV[v].begin(), VV[v].end());
        for (int v : VV[cut(i, 1)])
            VV[v] = VV[cut(i, 1)];
    }

    std::cout << "here" << std::endl;
    Vd angles_total(uv.rows()); angles_total.setConstant(0);
    for (int i = 0; i < angles_total.rows(); i++)
    {
        for (auto v : VV[i]) angles_total(i) += angles(v);
    }

    std::cout << "check total angles: " << std::endl;
    std::vector<bool> is_visited(uv.rows(), false);
    std::vector<std::vector<int>> cones;
    std::vector<double> cone_angles;
    for (int i = 0; i < angles_total.rows(); i++)
    {
        if (is_visited[i]) continue;
        if (std::abs(angles_total(i) - 2 * igl::PI) > 1e-5)
        {   
            std::vector<int> onecone;
            std::cout << "(";
            for (int v : VV[i]) 
            {
                std::cout << v << " ";
                is_visited[v] = true;
                onecone.push_back(v);
            }
            std::cout << ") : " << angles_total(i) / igl::PI * 180 << std::endl;
            cones.push_back(onecone);
            cone_angles.push_back(angles_total(i) / igl::PI * 180);
        }
    }

    for (int i = 0; i < Sn.rows(); i++)
    {
        if (Sn(i) != 0)
        {
            std::cout << i << "\t" << Sn(i) << std::endl;
        }
    }
    igl::serialize(cones, "cones", "./cones/cones_serialized_" + name, true);
    igl::serialize(cone_angles, "cone_angles", "./cones/cones_serialized_" + name);
}

int main(int argc, char *argv[])
{
    auto cmdl = argh::parser(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);
    if (cmdl[{"-h", "-help"}])
    {
        std::cout << "Usage: ./test_bin -options" << std::endl;
        std::cout << "-in: input model name" << std::endl;
        std::cout << "-o: output model name" << std::endl;
        // std::cout << "-uv: input uv" << std::endl;
        std::cout << "-c: opt steps" << std::endl;
        exit(0);
    }

    int threshold;
    std::string model, outfile;
    bool use_harm, space_filling_curve;
    int total_steps;
    cmdl("-in") >> model;
    // cmdl("-uv") >> uv_file;
    cmdl("-c") >> total_steps;
    cmdl("-o", model + "_out.obj") >> outfile;

    std::string name = model.substr(0, model.find_last_of("_"));
    name = name.substr(name.find_last_of('/') + 1);

    Eigen::MatrixXd V, uv, uv_test, uv_out;
    Eigen::MatrixXi F_uv, F;
    Eigen::VectorXi S, JJ;
    Vd SS;
    std::map<std::pair<int, int>, std::vector<int>> corres;
    igl::deserialize(V, "V", model); // output the original one
    igl::deserialize(F_uv, "Fuv", model);
    igl::deserialize(F, "F3d", model);
    // igl::deserialize(uv_test, "uv", model);
    igl::deserialize(uv, "uv", model);
    igl::deserialize(corres, "corres", model);
    igl::deserialize(SS, "S", model);
    igl::deserialize(JJ, "J", model);
    
    std::cout << "SS size:" << SS.rows() << std::endl;
    std::cout << "JJ size:" << JJ.rows() << std::endl;

    Eigen::VectorXd Sn(V.rows());
    Sn.setConstant(0);
    for (int i = 0; i < V.rows(); i++)
    {
       if (SS(JJ(i)) != 0) Sn(i) = 360 * (1 - SS(JJ(i)));
    }
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
    std::cout << corres.size() << std::endl;
    
    igl::writeOBJ(name+"_no_match.obj", V, F, uv, F_uv, uv, F_uv);

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
    std::cout << "check corres:" << std::endl;
    for (auto it = corres.begin(); it != corres.end(); it++)
    {
        std::cout << "(" << it->first.first << "," << it->first.second << "):(";
        for (int v : it->second)
            std::cout << v << ",";
        std::cout << ")" << std::endl;
    }


    /////////////////////////////////
    // prepare for match_maker
    ////////////////////////////////
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
    std::cout << R.rows() << " " << T.rows() << " " << polygon.rows() << std::endl;
    match_maker(V, F, uv_new, c, ci, R, T, polygon, mark);

    ///////////////////////////////////
    // update cut (constraints)
    //////////////////////////////////
    auto cut_copy = cut;
    for (int i = 0; i < polygon.rows(); i++)
    {
        for (int cut_row = 0; cut_row < cut.rows(); cut_row++)
        {
            for (int cut_col = 0; cut_col < 4; cut_col++)
            {
                if (cut_copy(cut_row, cut_col) == i)
                    cut(cut_row, cut_col) = T(i);
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

    // check_total_angle(uv_new, F, cut, Sn, name);
    // compute triangle areas
    Vd dblarea_uv;
    dblarea_uv *= 0.5;
    igl::doublearea(uv_new, F, dblarea_uv);
    Vd dblarea;
    igl::doublearea(V, F, dblarea);
    dblarea *= 0.5;
    double mesh_area = dblarea.sum();

    //////////////////////////
    // check triangle quality
    /////////////////////////
    std::cout << "check triangle quality:" << std::endl;
    double max_ar_uv = -1;
    double max_ar = -1;
    for (int i = 0; i < F.rows(); i++)
    {
        double max_l = -1, max_l_uv = -1;
        for (int j = 0; j < 3; j++)
        {
            int v0 = F(i, j), v1 = F(i, (j + 1) % 3);
            double l_uv = (uv_new.row(v0) - uv_new.row(v1)).norm();
            double l = (V.row(v0) - V.row(v1)).norm();
            if (l_uv > max_l_uv)
                max_l_uv = l_uv;
            if (l > max_l)
                max_l = l;
        }
        double ar_uv = max_l_uv * max_l_uv / 2 / dblarea_uv(i);
        double ar = max_l * max_l / 2 / dblarea(i);
        if (ar_uv > max_ar_uv)
            max_ar_uv = ar_uv;
        if (ar > max_ar)
            max_ar = ar;
    }
    std::cout << "max_ar_uv : " << max_ar_uv << std::endl;
    std::cout << "mar_ar : " << max_ar << std::endl;
    
    ///////////////////////////
    // build constraints
    spXd Aeq;
    buildAeq(cut, uv_new, F, Aeq);
    spXd AeqT = Aeq.transpose();

    spXd Dx, Dy, G;
    prepare(V, F, Dx, Dy);
    G = combine_Dx_Dy(Dx, Dy);

    auto compute_energy = [&G, &dblarea](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        return compute_energy_from_jacobian(Ji, dblarea);
    };

    auto compute_energy_max = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        auto E = symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
        double max_e = -1;
        for (int i = 0; i < E.size(); i++)
        {
            if (E(i) > max_e)
            {
                max_e = E(i);
            }
        }
        return max_e;
    };

    double energy = compute_energy(uv_new);
    std::cout << "Start Energy" << energy << std::endl;

    auto do_opt = [&F, &Aeq, &AeqT, &G, &dblarea, &compute_energy, &compute_energy_max](Eigen::MatrixXd &cur_uv, int N) {
        double energy = compute_energy(cur_uv);
        double energy_old = -1;
        clock_t iter_start_time, iter_end_time;
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
            iter_start_time = clock();
            
            solver.factorize(kkt);
            std::cout << "solver.info = " << solver.info() << std::endl;
            

            Vd newton = solver.solve(grad);

            iter_end_time = clock();
            std::cout << "time = " << (double)(iter_end_time - iter_start_time) / CLOCKS_PER_SEC << std::endl << std::endl;
            Vd res = kkt * newton - grad;
            std::cout << "residual:" << res.norm() << "\t" << std::max(res.maxCoeff(), -res.minCoeff()) << std::endl;
            
            Vd w = -newton.tail(newton.rows() - hessian.cols());
            newton.conservativeResize(hessian.cols());
            grad.conservativeResize(hessian.cols());
            Vd gradL = grad + AeqT * w;
            Xd new_dir = -Eigen::Map<Xd>(newton.data(), cur_uv.rows(), 2); // newton dir
            energy = bi_linesearch(F, cur_uv, new_dir, compute_energy, grad, energy);

            std::cout << std::setprecision(20) << "E_avg= " << energy << "\tE_max = " << compute_energy_max(cur_uv) << std::endl;
            std::cout << "grad.norm() = " << gradL.norm() << std::endl;
            
            iter_end_time = clock();
            std::cout << "time = " << (double)(iter_end_time - iter_start_time) / CLOCKS_PER_SEC << std::endl << std::endl;
            if (energy == energy_old)
            {
                std::cout << "opt finished" << std::endl;
                break;
            }
            energy_old = energy;
        }
        return energy;
    };
    // return 0;

    while (total_steps > 0)
    {
        do_opt(uv_new, 5);
        total_steps -= 5;
        igl::serialize(V, "V", name + "_latest_serialized", true);
        igl::serialize(uv_new, "uv", name + "_latest_serialized");
        igl::serialize(cut, "cut", name + "_latest_serialized");
        igl::serialize(F, "F", name + "_latest_serialized");
    }

    double lendiff_max = -1;
    for (int i = 0; i < cut.rows(); i++)
    {
        double l1 = (uv_new.row(cut(i, 0)) - uv_new.row(cut(i, 1))).norm();
        double l2 = (uv_new.row(cut(i, 2)) - uv_new.row(cut(i, 3))).norm();
        if (fabs(l1 - l2) > 1e-5)
        {
            std::cout << cut.row(i) << "\tlendiff: " << fabs(l1 - l2) << std::endl;
        }
        if (fabs(l1 - l2) > lendiff_max)
        {
            lendiff_max = fabs(l1 - l2);
        }
    }
    std::cout << "max lendiff : " << lendiff_max << std::endl;
    buildAeq(cut, uv_new, F, Aeq);
    std::cout << "#fl= " << check_flip(uv_new, F) << std::endl;
    Eigen::MatrixXd CN;
    Eigen::MatrixXi FN;
    igl::writeOBJ(outfile, V, F, uv_new, F, uv_new, F);
}
