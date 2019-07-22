// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hello_test.cu
 *
 * @brief Test related functions for hello
 */

#pragma once

// Boost includes for CPU CC reference algorithms
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

namespace gunrock {
namespace app {
namespace cc {


/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference hello ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(
    const GraphT &graph,
    typename GraphT::VertexT *component_labels,
    bool quiet)
{
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::CsrT CsrT;

    auto &row_offsets = graph.CsrT::row_offsets;
    auto &column_indices = graph.CsrT::column_indices;
    SizeT num_nodes = graph.nodes;

    //
    // Copy our graph's data into something suitable for Boost
    //
    using namespace boost;
    adjacency_list<vecS, vecS, undirectedS> boost_graph;

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
            add_edge(i, column_indices[j], boost_graph);
        }
    }

    //
    // Run and time Boost Graph's connected components
    //
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    size_t num_components = connected_components(boost_graph, &component_labels[0]); 
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    if (!quiet) {
        printf("CPU CC finished in %lf msec. Found %zu components\n", elapsed, num_components);
    }

    return elapsed;
}

/**
 * @brief Validation of hello results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(
             util::Parameters &parameters,
             GraphT           &graph,
             typename GraphT::VertexT *h_component_ids,
             typename GraphT::VertexT *ref_component_labels,
             // </TODO>
             bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    // <TODO> result validation and display
    // for(SizeT v = 0; v < graph.nodes; ++v) {
    //     printf("%d %f %f\n", v, h_degrees[v], ref_degrees[v]);
    // }
    // </TODO>

    if(num_errors == 0) {
       util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    }

    return num_errors;
}

} // namespace Template
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
