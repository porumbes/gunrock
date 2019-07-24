// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * Template_enactor.cuh
 *
 * @brief hello Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>
 
#include <gunrock/app/cc/cc_problem.cuh>
// </TODO>


namespace gunrock {
namespace app {
namespace cc {
// </TODO>

/**
 * @brief Speciflying parameters for hello Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(app::UseParameters_enactor(parameters));

    // <TODO> if needed, add command line parameters used by the enactor here
    // </TODO>
    
    return retval;
}

/**
 * @brief defination of hello iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct CCIterationLoop : public IterationLoopBase
    <EnactorT, Use_FullQ | Push
    // <TODO>if needed, stack more option, e.g.:
    // | (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
    // Update_Predecessors : 0x0)
    // </TODO>
    >
{
    typedef typename EnactorT::VertexT VertexT;
    typedef typename EnactorT::SizeT   SizeT;
    typedef typename EnactorT::ValueT  ValueT;
    typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
    typedef typename EnactorT::Problem::GraphT::GpT  GpT;
    
    typedef IterationLoopBase
        <EnactorT, Use_FullQ | Push
        // <TODO> add the same options as in template parameters here, e.g.:
        // | (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
        // Update_Predecessors : 0x0)
        // </TODO>
        > BaseIterationLoop;

    CCIterationLoop() : BaseIterationLoop() {}

    /**
     * @brief Core computation of hello, one iteration
     * @param[in] peer_ Which GPU peers to work on, 0 means local
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Core(int peer_ = 0)
    {
        // --
        // Alias variables
        
        auto &data_slice = this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        
        auto &enactor_slice = this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        
        auto &enactor_stats    = enactor_slice.enactor_stats;
        auto &graph            = data_slice.sub_graph[0];
        auto &oprtr_parameters = enactor_slice.oprtr_parameters;
        auto &retval           = enactor_stats.retval;
        auto &iteration        = enactor_stats.iteration;
        
        // CC specific problem data        
        auto &component_ids = data_slice.component_ids;
        auto &vertex_flag   = data_slice.vertex_flag;
        auto &edge_flag     = data_slice.edge_flag;
        auto &marks         = data_slice.marks;
        auto &masks         = data_slice.masks;
        auto &froms         = data_slice.froms;
        auto &tos           = data_slice.tos; 
        // </TODO>
        
        //
        // Hook Initialization
        //
        auto hook_init_op = [
            tos,
            component_ids
        ] __host__ __device__ (VertexT *froms_, const SizeT &id)
        {
            VertexT from_node = froms_[id];
            VertexT to_node   = tos[id];

            VertexT max_node = from_node > to_node ? from_node : to_node;
            VertexT min_node = from_node + to_node - max_node;

            component_ids[max_node] = min_node;
        };

        printf("froms.GetSize = %d, tos.GetSize = %d\n", froms.GetSize(), tos.GetSize());
        // for every edge
        GUARD_CU(froms.ForAll(hook_init_op,
                             graph.edges,
                             util::DEVICE,
                             oprtr_parameters.stream));
        GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");
        printf("After froms.ForAll\n");

        //
        // Pointer Jumping
        //
        vertex_flag[0] = 0;
        while(!vertex_flag[0]) 
        {
            printf("in while(!vertex_flag[0])\n");
            vertex_flag[0] = 1;
            GUARD_CU(vertex_flag.Move(util::HOST, util::DEVICE));

            auto ptr_jump_op = [ 
                vertex_flag 
            ] __host__ __device__ (VertexT *component_ids_, const SizeT &id) 
            {
                VertexT parent         = Load<cub::LOAD_CG>(component_ids_ + id);
                VertexT grand_parent   = Load<cub::LOAD_CG>(component_ids_ + parent);

                if (parent != grand_parent) 
                {
                    Store(vertex_flag + 0, 0);
                    Store(component_ids_ + id, grand_parent);
                }
            };

            printf("component_ids.GetSize = %d\n", component_ids.GetSize());
            GUARD_CU(component_ids.ForAll(ptr_jump_op,
                                          graph.nodes,
                                          util::DEVICE,
                                          oprtr_parameters.stream));

            // SDP not sure what to do about this type of stuff
            // what does queue_reset = false actually do?
            // frontier_attribute->queue_reset = false;
            // frontier_attribute->queue_index++;
            // enactor_stats->iteration++;

            // ptr_jump_op could have modified vertex_flag,
            // move it back to the HOST
            GUARD_CU(vertex_flag.Move(util::DEVICE, util::HOST));

            // SDP, not sure why we need to (if we need to) synchronize here?
            // Could be because of the Move directly above, but then why didn't we
            // sync with the Move from HOST to DEVICE at the top of the while loop?
            // Old API does this.
            GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");
        }
        
        // SDP, not really sure what this is about.
        // Should this be more of a stop condition? An optimization?
        // if (iteration > 1 && 
        //     (graph.edges / 3 > graph.nodes))
        // {
        //     printf("returning iteration = %d\n", iteration);
        //     return cudaSuccess;
        // }

        // 
        // Prepare for Update Mask
        //
        printf("preparing for Update Mask\n");
        GUARD_CU(marks.ForEach([]__host__ __device__ (bool &mark){
            mark = false;
        }, graph.edges, util::DEVICE, oprtr_parameters.stream));

        // SDP, figure out the following:
        // frontier_attribute->queue_index   = 0;        // Work queue index
        // frontier_attribute->selector      = 0;
        // frontier_attribute->queue_length  = graph_slice->nodes;
        // frontier_attribute->queue_reset   = true;


        //
        // Update Mask
        //
        // SDP, SizeT vs VertexID? Does it matter? See RunCC in old api,
        // all template parameters are <int, int, int>. Simple answer is yes
        // in general. Should we be casting? What happens if VertexID is a float?
        // Is that even possible for CC? 
        auto update_mask_op = [ // SDP, masks_ tracks if the given id is a root node
            component_ids
        ] __host__ __device__ (signed char *masks_, const SizeT &id)
        {
            VertexT parent = Load<cub::LOAD_CG>(component_ids + id);
            masks_[id] = (parent == id) ? 0 : 1;
        };

        printf("update_mask_op\n");
        GUARD_CU(masks.ForAll(update_mask_op,
                              graph.nodes,
                              util::DEVICE,
                              oprtr_parameters.stream) );
        printf("After update_mask_op\n");

        // SDP, figure out how to incorporate the following:
        // if (enactor -> debug && (enactor_stats->retval = 
        //     util::GRError("filter::Kernel Update Mask Operation failed", 
        //         __FILE__, __LINE__))) return;
        // enactor_stats -> nodes_queued[0] += frontier_attribute->queue_length;        

        //
        // Hooking and additional Pointer Jumping
        //
        // enactor_stats->iteration = 1; // SDP, figure out how to incorporate
        edge_flag[0] = 0;
        while (!edge_flag[0])
        {
            printf("in while (!edge_flag[0]) \n");
            //
            // Prepare for Hook Max
            //
            // SDP, figure out how to incorporate
            // frontier_attribute->queue_index  = 0;        // Work queue index
            // frontier_attribute -> queue_length = graph_slice -> edges;
            // frontier_attribute->selector     = 0;
            // frontier_attribute->queue_reset  = true;

            edge_flag[0] = 1;
            GUARD_CU(edge_flag.Move(util::HOST, util::DEVICE)); 

            //
            // Hook Max
            //
            auto hook_max_op = [
                component_ids,
                froms,
                tos,
                edge_flag
            ] __host__ __device__ (bool *marks_, const SizeT &id)
            {
                bool mark = Load<cub::LOAD_CG>(marks_ + id);
                if (!mark)
                {
                    VertexT from_node      = Load<cub::LOAD_CG>(froms + id);
                    VertexT to_node        = Load<cub::LOAD_CG>(tos + id);
                    VertexT parent_from    = Load<cub::LOAD_CG>(component_ids + from_node);
                    VertexT parent_to      = Load<cub::LOAD_CG>(component_ids + to_node);  

                    if (parent_from == parent_to)
                    {
                        Store(marks_ + id, true);
                    } 
                    else 
                    { 
                        VertexT max_node = parent_from > parent_to ? parent_from : parent_to;
                        VertexT min_node = parent_from + parent_to - max_node;
                        Store(component_ids + max_node, min_node);
                        Store(edge_flag + 0, 0);
                    }                  
                }

            };

            GUARD_CU(marks.ForAll(hook_max_op,
                                  graph.edges,
                                  util::DEVICE,
                                  oprtr_parameters.stream));

            // hook_max_op could have modified edge_flag,
            // move it back to the HOST
            GUARD_CU(edge_flag.Move(util::DEVICE, util::HOST));

            // SDP, not sure why we need to (if we need to) synchronize here?
            // Could be because of the Move directly above, but then why didn't we
            // sync with the Move from HOST to DEVICE at the top of the while loop?
            // Old API does this.
            GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");


            // SDP, figure out how to incorporate the following:
            // if (enactor -> debug && (enactor_stats->retval = 
            //     util::GRError("filter::Kernel Hook Min/Max Operation failed", 
            //         __FILE__, __LINE__))) return;
            // enactor_stats -> edges_queued[0] += frontier_attribute->queue_length;
            // frontier_attribute->queue_reset = false;
            // frontier_attribute->queue_index++;
            // enactor_stats->iteration++;

            // Check if done
            if (edge_flag[0]) break; //|| enactor_stats->iteration>5) break;
            }
            //
            // Prepare for Pointer Jump Mask
            //
            // SDP, figure out how to incorporate
            // frontier_attribute->queue_index  = 0;
            // frontier_attribute->selector     = 0;
            // frontier_attribute->queue_length = graph_slice->nodes;
            // frontier_attribute->queue_reset  = true;

            //
            // Pointer Jump Mask
            //
            vertex_flag[0] = 0;
            while (!vertex_flag[0])
            {
                printf("In Pointer Jump Mask -- while (!vertex_flag[0])\n");
                vertex_flag[0] = 1;
                GUARD_CU(vertex_flag.Move(util::HOST, util::DEVICE)); 

                auto ptr_jump_mask_op = [
                    masks,
                    vertex_flag
                ] __host__ __device__ (VertexT *component_ids_, const SizeT &id)
                {
                    if (masks[id] == 0)
                    {
                        VertexT parent         = Load<cub::LOAD_CG>(component_ids_ + id);
                        VertexT grand_parent   = Load<cub::LOAD_CG>(component_ids_ + parent);

                        if (parent != grand_parent) 
                        {
                            vertex_flag[0] = 0; // SDP why was Store not used in Functor version for this? Was used in PtrJumpIterationLoop.
                            Store(component_ids_ + id, grand_parent);
                        } 
                        else 
                        {
                            masks[id] = -1; // SDP, should this be a Store too?
                        }
                    }
                };

                GUARD_CU(component_ids.ForAll(ptr_jump_mask_op,
                                              graph.nodes,
                                              util::DEVICE,
                                              oprtr_parameters.stream));

                // ptr_jump_mask_op could have modified vertex_flag,
                // move it back to the HOST
                GUARD_CU(vertex_flag.Move(util::DEVICE, util::HOST));

                // SDP, not sure why we need to (if we need to) synchronize here?
                // Could be because of the Move directly above, but then why didn't we
                // sync with the Move from HOST to DEVICE at the top of the while loop?
                // Old API does this.
                GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream), "cudaStreamSynchronize failed");

                // SDP, figure out how to incorporate
                // enactor_stats -> nodes_queued[0] += frontier_attribute->queue_length;
                // frontier_attribute->queue_reset = false;
                // frontier_attribute->queue_index++;
            }

            //
            // Pointer Jump Unmask
            //

            // SDP, figure out how to incorporate the following:
            // frontier_attribute->queue_index  = 0;        // Work queue index
            // frontier_attribute->selector     = 0;
            // frontier_attribute->queue_length = graph_slice->nodes;
            // frontier_attribute->queue_reset  = true;

            auto ptr_jump_unmask = [
                masks
            ]  __host__ __device__ (VertexT *component_ids_, const SizeT &id)
            {
                if (masks[id] == 1) 
                {
                    VertexT parent         = Load<cub::LOAD_CG>(component_ids_ + id);
                    VertexT grand_parent   = Load<cub::LOAD_CG>(component_ids_ + parent);

                    Store(component_ids_ + id, grand_parent);
                }
            };

            GUARD_CU(component_ids.ForAll(ptr_jump_unmask,
                                          graph.nodes,
                                          util::DEVICE,
                                          oprtr_parameters.stream));

            // SDP, figure out how to incorporate
            // enactor_stats -> nodes_queued[0] += frontier_attribute->queue_length;

            //
            // Update Mask
            // 
            GUARD_CU(masks.ForAll(update_mask_op,
                                 graph.nodes,
                                 util::DEVICE,
                                 oprtr_parameters.stream));

            // SDP, figure out how to incorporate
            // enactor_stats -> nodes_queued[0] += frontier_attribute->queue_length;
        }

        // SDP, figure out how to incorporate
        // enactor_stats->iteration = data_slice->turn;
        
        return retval;
    }

    /**
     * @brief Routine to combine received data and local data
     * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each transmition item, typed VertexT
     * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each transmition item, typed ValueT
     * @param  received_length The numver of transmition items received
     * @param[in] peer_ which peer GPU the data came from
     * \return cudaError_t error message(s), if any
     */
    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    cudaError_t ExpandIncoming(SizeT &received_length, int peer_)
    {
        
        // ================ INCOMPLETE TEMPLATE - MULTIGPU ====================
        
        auto &data_slice    = this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto &enactor_slice = this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        //auto iteration = enactor_slice.enactor_stats.iteration;
        // TODO: add problem specific data alias here, e.g.:
        // auto         &distances          =   data_slice.distances;

        auto expand_op = [
        // TODO: pass data used by the lambda, e.g.:
        // distances
        ] __host__ __device__(
            VertexT &key, const SizeT &in_pos,
            VertexT *vertex_associate_ins,
            ValueT  *value__associate_ins) -> bool
        {
            // TODO: fill in the lambda to combine received and local data, e.g.:
            // ValueT in_val  = value__associate_ins[in_pos];
            // ValueT old_val = atomicMin(distances + key, in_val);
            // if (old_val <= in_val)
            //     return false;
            return true;
        };

        cudaError_t retval = BaseIterationLoop:: template ExpandIncomingBase
            <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            (received_length, peer_, expand_op);
        return retval;
    }

    bool Stop_Condition(int gpu_num = 0)
    {
        auto &data_slice = this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
            
        auto &vertex_flag   = data_slice.vertex_flag;
        auto &edge_flag     = data_slice.edge_flag;

        printf("Stop_Condition: vf = %d, ef = %d\n", vertex_flag[0], edge_flag[0]);

        return vertex_flag[0] == 1 && 
               edge_flag[0] == 1;
    }
}; // end of CCIterationLoop

/**
 * @brief Template enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <
    typename _Problem,
    util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor :
    public EnactorBase<
        typename _Problem::GraphT,
        typename _Problem::GraphT::VertexT,
        typename _Problem::GraphT::ValueT,
        ARRAY_FLAG, cudaHostRegisterFlag>
{
public:
    typedef _Problem                   Problem ;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexT  VertexT ;
    typedef typename Problem::GraphT   GraphT  ;
    typedef typename GraphT::VertexT   LabelT  ;
    typedef typename GraphT::ValueT    ValueT  ;
    typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
        BaseEnactor;
    typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> 
        EnactorT;
    typedef CCIterationLoop<EnactorT> 
        IterationT;

    Problem *problem;
    IterationT *iterations;

    /**
     * @brief hello constructor
     */
    Enactor() :
        BaseEnactor("Template"),
        problem    (NULL  )
    {
        // <TODO> change according to algorithmic needs
        this -> max_num_vertex_associates = 0;
        this -> max_num_value__associates = 1;
        // </TODO>
    }

    /**
     * @brief hello destructor
     */
    virtual ~Enactor() { /*Release();*/ }

    /*
     * @brief Releasing allocated memory space
     * @param target The location to release memory from
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Release(target));
        delete []iterations; iterations = NULL;
        problem = NULL;
        return retval;
    }

    /**
     * @brief Initialize the problem.
     * @param[in] problem The problem object.
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Init(
        Problem          &problem,
        util::Location    target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        this->problem = &problem;

        // Lazy initialization
        GUARD_CU(BaseEnactor::Init(
            problem, Enactor_None,
            // <TODO> change to how many frontier queues, and their types
            2, NULL,
            // </TODO>
            target, false));
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++) {
            GUARD_CU(util::SetDevice(this -> gpu_idx[gpu]));
            auto &enactor_slice = this -> enactor_slices[gpu * this -> num_gpus + 0];
            auto &graph = problem.sub_graphs[gpu];
            GUARD_CU(enactor_slice.frontier.Allocate(
                graph.nodes, graph.edges, this -> queue_factors));
        }

        iterations = new IterationT[this -> num_gpus];
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++) {
            GUARD_CU(iterations[gpu].Init(this, gpu));
        }

        GUARD_CU(this -> Init_Threads(this,
            (CUT_THREADROUTINE)&(GunrockThread<EnactorT>)));
        return retval;
    }

    /**
      * @brief one run of hello, to be called within GunrockThread
      * @param thread_data Data for the CPU thread
      * \return cudaError_t error message(s), if any
      */
    cudaError_t Run(ThreadSlice &thread_data)
    {
        gunrock::app::Iteration_Loop<
            // <TODO> change to how many {VertexT, ValueT} data need to communicate
            //       per element in the inter-GPU sub-frontiers
            0, 1,
            // </TODO>
            IterationT>(
            thread_data, iterations[thread_data.thread_num]);
        return cudaSuccess;
    }

    /**
     * @brief Reset enactor
...
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Reset(
        // <TODO> problem specific data if necessary, eg
        VertexT src = 0,
        // </TODO>
        util::Location target = util::DEVICE)
    {
        typedef typename GraphT::GpT GpT;
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Reset(target));

        // <TODO> Initialize frontiers according to the algorithm:
        // In this case, we add a single `src` to the frontier
        for (int gpu = 0; gpu < this->num_gpus; gpu++) {
           if ((this->num_gpus == 1) ||
                (gpu == this->problem->org_graph->GpT::partition_table[src])) {
               this -> thread_slices[gpu].init_size = 1;
               for (int peer_ = 0; peer_ < this -> num_gpus; peer_++) {
                   auto &frontier = this -> enactor_slices[gpu * this -> num_gpus + peer_].frontier;
                   frontier.queue_length = (peer_ == 0) ? 1 : 0;
                   if (peer_ == 0) {
                       GUARD_CU(frontier.V_Q() -> ForEach(
                           [src]__host__ __device__ (VertexT &v) {
                           v = src;
                       }, 1, target, 0));
                   }
               }
           } else {
                this -> thread_slices[gpu].init_size = 0;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++) {
                    this -> enactor_slices[gpu * this -> num_gpus + peer_].frontier.queue_length = 0;
                }
           }
        }
        // </TODO>
        
        GUARD_CU(BaseEnactor::Sync());
        return retval;
    }

    /**
     * @brief Enacts a hello computing on the specified graph.
...
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact(
        // <TODO> problem specific data if necessary, eg
        VertexT src = 0
        // </TODO>
    )
    {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(this -> Run_Threads(this));
        util::PrintMsg("GPU Template Done.", this -> flag & Debug);
        return retval;
    }
};

} // namespace Template
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
