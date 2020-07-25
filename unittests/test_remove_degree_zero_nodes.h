/*
 * @brief Test modifying a graph by removing all degree zero nodes.
 * @file test_remove_degree_zero_nodes.h
 */

// Utilities and correctness-checking
//#include <gunrock/util/test_utils.h>

#include <gunrock/util/parameters.h>
//#include <gunrock/util/error_utils.cuh>



//#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph defintions
#include <gunrock/graphio/graphio.cuh>

#include <gunrock/graphio/utils.cuh>

//#include <gunrock/app/app_base.cuh>

//#include <gunrock/app/test_base.cuh>




#include<iostream>
#include<fstream>

struct a_test_main_struct {

    static int test_num_nodes;

    /**
     * @brief the actual main function, after type switching
     * @tparam VertexT    Type of vertex identifier
     * @tparam SizeT      Type of graph size, i.e. type of edge identifier
     * @tparam ValueT     Type of edge values
     * @param  parameters Command line parameters
     * @param  v,s,val    Place holders for type deduction
     * \return cudaError_t error message(s), if any
     */
    template <
        typename VertexT, // Use int as the vertex identifier
        typename SizeT,   // Use int as the graph size type
        typename ValueT>  // Use int as the value type
    cudaError_t operator()(gunrock::util::Parameters &parameters,
                           VertexT v, SizeT s, ValueT val) {

        using namespace gunrock;

        // typedef typename app::TestGraph<VertexT, SizeT, ValueT,
        //     graph::HAS_EDGE_VALUES | graph::HAS_COO | graph::HAS_CSR | graph::HAS_CSC>
        //     GraphT;
        using GraphT = app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR>;
            
        cudaError_t retval = cudaSuccess;
        util::CpuTimer cpu_timer;
        GraphT graph;

        cpu_timer.Start();
        GUARD_CU(graphio::LoadGraph(parameters, graph));
        cpu_timer.Stop();
        parameters.Set("load-time", cpu_timer.ElapsedMillis());

        std::string my_str;
        GUARD_CU(parameters.Get("load-time", my_str));
        std::cout << "load-time = " << my_str << "\n";
        
       
        graphio::RemoveDegreeZeroNodes(graph);
        test_num_nodes = graph.nodes;

        return retval;
    }
};

int a_test_main_struct::test_num_nodes = 0;

TEST(utils, RemoveZeroDegreeNodes) {

    //
    // Prepare our Parameters
    //

    // easiest way is to fake a command line
    char* const argv[] = {
        "not_used", // normally the name of the executable
        /*"--graph-type",*/ "market",
        // Assumes executable is being run from within build directory
        // like: ./bin/unittests
        // Not sure if there's a "nice" way of easily getting the path
        // based on `unittest` executable.
        "../dataset/small/bips98_606.mtx",
        // SDP figure out what this is actually removing? "--remove-self-loops=true" 
    };
    const int argc = sizeof(argv) / sizeof(char*); // number of string pointers in argv

    using namespace gunrock;

    gunrock::util::Parameters parameters("GoogleTest RemoveZeroDegreeNodes");

    auto retval = gunrock::graphio::UseParameters(parameters);
    ASSERT_EQ(retval, cudaSuccess);

    retval = gunrock::app::UseParameters_test(parameters);
    ASSERT_EQ(retval, cudaSuccess);

    retval = gunrock::app::UseParameters_app(parameters);
    ASSERT_EQ(retval, cudaSuccess);

    retval = parameters.Parse_CommandLine(argc, argv);
    ASSERT_EQ(retval, cudaSuccess);

    retval = parameters.Check_Required();
    ASSERT_EQ(retval, cudaSuccess);

        
    //
    // Do the heavy lifting of preparing a Graph Type
    // and loading it.
    //

    retval = app::Switch_Types<
                app::VERTEXT_U32B   | //app::VERTEXT_U64B   |
                app::SIZET_U32B     | //app::SIZET_U64B     |
                app::VALUET_F32B    | //app::VALUET_F64B    |  
                app::UNDIRECTED     //| app::DIRECTED        
                >
                (parameters, a_test_main_struct());

    // verify number of resulting nodes verus known number
    ASSERT_EQ(6594, a_test_main_struct::test_num_nodes);

    // for(int i = 0; i < argc; i++)
    // {
    //     std::cout << "arg[" << i << "] = " << argv[i] << "\n";
    // }

    // std::ifstream graph_file(argv[3]);
    // if(graph_file.is_open()) {

    //     std::cout << "opened: " << argv[2] << "\n";
    //     graph_file.close();
    // }
    // else
    // {
    //     std::cout << "couln't open file!" << "\n";
    // }
    


    std::cout << parameters.Get_CommandLine() << "\n";
    // prepare a Graph Type

    // load a known file from our data directory
    //GUARD_CU(graphio::LoadGraph(parameters, graph));

    // RemoveStandaloneNodes

    // verify number of nodes verus known number
}
