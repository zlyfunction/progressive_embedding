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
long global_autodiff_time = 0;
long global_project_time = 0;

int main(int argc, char *argv[])
{
    auto cmdl = argh::parser(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);
    if (cmdl[{"-h", "-help"}])
    {
        std::cout << "Usage: ./test_bin -options" << std::endl;
        std::cout << "-in: input model name" << std::endl;
        std::cout << "-o: output model name" << std::endl;
        std::cout << "-uv: input uv" << std::endl;
        exit(0);
    }

    int threshold;
    std::string model, uv_file, outfile;
    bool use_bd, space_filling_curve;
    cmdl("-in") >> model;
    cmdl("-uv") >> uv_file;
    cmdl("-o", model + "_out.obj") >> outfile;

    Eigen::MatrixXd V, uv, uv_out;
    Eigen::MatrixXi F_uv, F;
    Eigen::VectorXd S;
    std::map<std::pair<int, int>, std::vector<int>> corres;
    igl::deserialize(V, "V", model); // output the original one
    igl::deserialize(F_uv, "Fuv", model);
    igl::deserialize(F, "F_3d", model);
    // igl::deserialize(uv, "uv", model);
    igl::deserialize(uv, "cur_uv", uv_file);
    igl::deserialize(S, "S", model);
    igl::deserialize(corres, "corres", model);

    std::cout << V.rows() << std::endl;
    std::cout << F_uv.rows() << std::endl;
    std::cout << F.rows() << std::endl;
    std::cout << uv.rows() << std::endl;
    std::cout << S.rows() << std::endl;
    std::cout << corres.size() << std::endl;

    std::vector<std::vector<int>> bds_uv;
    igl::boundary_loop(F_uv, bds_uv);

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
                R(size0 + j) = floor(1 - S(v1));
            else
                R(size0 + j) = 0;
            if (j == 0)
                mark(size0 + j) = 1;
            else
                mark(size0 + j) = 0;
        }
    }
    std::cout << R.rows() << " " << T.rows() << " " << polygon.rows();
    
    match_maker(V, F, uv_new, c, ci, R, T, polygon, mark);

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
            std::cout << E.maxCoeff() << std::endl;
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
    vr.callback_key_down = key_down;
    //plot_mesh(vr,uv,F,{},Eigen::VectorXi());
    vr.launch();

    Eigen::MatrixXd CN;
    Eigen::MatrixXi FN;
    igl::writeOBJ(outfile, V, F, CN, FN, uv_new, F);
}
