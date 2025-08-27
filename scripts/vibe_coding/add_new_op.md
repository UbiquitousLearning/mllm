You are a programmer proficient in C++ and Python, and you are now required to strictly follow my instructions. Your overall task is to add a new operator, whose backend is {{__backend__}} and whose operator name is {{__op_name__}}.

The task overview consists of the following parts:

1. Create operator definition files (.hpp and .cpp)  
2. Implement the operator in the specified backend  
3. Register the operator factory  
4. Update the RTTI generation script and execute it  
5. Add IR support  

Detailed task descriptions are as follows:

1. **Create operator abstract definition file**  
   Create a file named {{__op_name__}}Op.hpp under the directory `mllm/core/aops/`, defining the abstract class of the operator:  
   - Inherit from the base class `BaseOp`  
   - Define the operator options struct `{{__op_name__}}OpOptions` (inheriting from `BaseOpOptions<{{__op_name__}}OpOptions>`)  
   - Declare all methods that need to be overridden: `load`, `trace`, `forward`, `reshape`, and `setup`

2. **Create operator implementation file**  
   Create a file named {{__op_name__}}Op.cpp under the directory `mllm/core/aops/`, implementing basic functionality of the abstract operator:  
   - Implement the constructor  
   - Implement the various virtual methods (you may provide default implementations as needed)

3. **Implement the concrete operator in the specified backend**  
   Create {{__op_name__}}Op.hpp and {{__op_name__}}Op.cpp files under `mllm/backends/{{__backend__}}/ops/`:  
   - In {{__op_name__}}Op.hpp, define class `CP{{__op_name__}}Op` inheriting from `aops::{{__op_name__}}Op`  
   - Define the operator factory class `{{__backend__}}{{__op_name__}}OpFactory`, inheriting from `TypedOpFactory<OpTypes::k{{__op_name__}}, aops::{{__op_name__}}OpOptions>`  
   - Implement the `createOpImpl` method within the factory class
   - Register the operator factory in `mllm/backends/{{__backend__}}/ops/{{__backend__}}.cpp`

4. **Update the RTTI generation script**  
   Add the new operator in the file `mllm/compile/ir/rtti_kind_gen.py`:  
   - Locate the `define_lianlg_ir` function  
   - Find the appropriate place within this function to add the new operator definition in the format: `op.derive(Cls("{{__op_name__}}Op"))`  
   - After adding, run the script to regenerate the RTTI-related files

5. **Add IR (Intermediate Representation) support**  
   Add the new operator definition in `mllm/compile/ir/linalg/Op.hpp`:  
   - Use the `LINALG_AOPS_DEFINE` macro to define the new IR operator class  
   - At the end of the file, add a new operator declaration in the format: `LINALG_AOPS_DEFINE({{__op_name__}}Op, {{__op_name__}}OP)`  
   - If implementation in a cpp file is required, also add the `LINALG_AOPS_DECL` macro in `mllm/compile/ir/linalg/Op.cpp`

**Important Notes:**

1. You are **not** required to implement the `forward` part of the operator — you are not proficient in this, and you should not attempt to write it!  
2. All files must adhere to the project's existing code style and architectural patterns  
3. Ensure proper inclusion of necessary header files  
4. The operator type must be defined in the `OpTypes` enum, but must only be inserted at the end, before `kOpType_End`  
5. The new operator must be registered following the project's factory pattern
6. Please check `./build_mllm.md` to learn how to build mllm.
