## Tensorflow C++

### 1. 自定义操作

在 TF 的官方文档中，定义一个操作，需要我们调用 `REGSITER_OP` 宏来注册。

```c++
REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

那么这个宏到底是什么那？下面就根据源文件进行下 Crash.

`/tensorflow/core/framework/op.h`.

```c++
#define REGISTER_OP(name) REGISTER_OP_UNIQ_HELPER(__COUNTER__, name)
#define REGISTER_OP_UNIQ_HELPER(ctr, name) REGISTER_OP_UNIQ(ctr, name)
#define REGISTER_OP_UNIQ(ctr, name)                                          \
  static ::tensorflow::register_op::OpDefBuilderReceiver register_op##ctr    \
      TF_ATTRIBUTE_UNUSED =                                                  \
          ::tensorflow::register_op::OpDefBuilderWrapper<SHOULD_REGISTER_OP( \
              name)>(name)
```

要理解上面的宏，需要的知识点：

- `\` 代码换行
- `##` 字符连接
- `TF_ATTRIBUTE_UNUSED` 其是 `#define TF_ATTRIBUTE_UNUSED __attribute__((unused))` 的宏定义。
- `__COUNTER__` 预处理器定义的自增量，遇到就会加一。

那么 `REGISTER_OP("ZeroOut")` 就会被预处理为：

```c++
static ::tensorflow::register_op::OpDefBuilderReceiver register_op1 __attribute__((unused)) = ::tensorflow::register_op::OpDefBuilderWrapper<true>("ZeroOut");
```

让我们再简化上面的声明语句:

```c++
static OpDefBuilderReceiver register_op1 = OpDefBuilderWrapper<true>("ZeroOut");
```

现在出现了 `OpDefBuilderReceiver` 和 `OpDefBuilderWrapper` 两个类型。那让我们从 `OpDefBuilderWrapper` 开始 crash.

#### 1.1 `OpDefBuilderWrapper`

从 `Wrapper` 这个后缀可以看出其是一包裹类 `/tensorflow/core/framework/op.h` 。

```c++
// OpDefBuilderWrapper is a templated class that is used in the REGISTER_OP
// calls. This allows the result of REGISTER_OP to be used in chaining, as in
// REGISTER_OP(a).Attr("...").Input("...");, while still allowing selective
// registration to turn the entire call-chain into a no-op.
template <bool should_register>
class OpDefBuilderWrapper;

// Template specialization that forwards all calls to the contained builder.
template <>
class OpDefBuilderWrapper<true> {
   public:
    explicit OpDefBuilderWrapper(const char name[]) : builder_(name) {}
    OpDefBuilderWrapper<true>& Attr(string spec) {
      builder_.Attr(std::move(spec));
      return *this;
    }
    OpDefBuilderWrapper<true>& Input(string spec) {
      builder_.Input(std::move(spec));
      return *this;
    }
    OpDefBuilderWrapper<true>& Output(string spec) {
      builder_.Output(std::move(spec));
      return *this;
    }
    OpDefBuilderWrapper<true>& SetIsCommutative() {
      builder_.SetIsCommutative();
      return *this;
    }
    OpDefBuilderWrapper<true>& SetIsAggregate() {
      builder_.SetIsAggregate();
      return *this;
    }
    OpDefBuilderWrapper<true>& SetIsStateful() {
      builder_.SetIsStateful();
      return *this;
    }
    OpDefBuilderWrapper<true>& SetAllowsUninitializedInput() {
      builder_.SetAllowsUninitializedInput();
      return *this;
    }
    OpDefBuilderWrapper<true>& Deprecated(int version, string explanation) {
      builder_.Deprecated(version, std::move(explanation));
      return *this;
    }
    OpDefBuilderWrapper<true>& Doc(string text) {
      builder_.Doc(std::move(text));
      return *this;
    }
    OpDefBuilderWrapper<true>& SetShapeFn(
        Status (*fn)(shape_inference::InferenceContext*)) {
      builder_.SetShapeFn(fn);
      return *this;
    }
    const ::tensorflow::OpDefBuilder& builder() const { return builder_; }

   private:
    mutable ::tensorflow::OpDefBuilder builder_;
}
```

从文件的注释中我们也看到了，该包裹类的上的就是了让其包裹的 `OpDefBuilder builder_` 通过以链式的写法调用。实现也很方便就是将所有调用传递给 builder_ 调用，然后返回 `*this` 就行了。

`class OpDefBuilderWrapper<true>` 对应的还有一个 `class OpDefBuilderWrapper<false>` 两者的区别就是会不会去完成注册。

那目前为止，我们包裹的 `OpDefBuidler` 类是做什么的那？该类定义在 `tensorflow/core/framework/op_def_builder.h`。

```c++
// Builder class passed to the REGISTER_OP() macro.
class OpDefBuilder {
 public:
  // Constructs an OpDef with just the name field set.
  explicit OpDefBuilder(string op_name);

  // Adds an attr to this OpDefBuilder (and returns *this). The spec has
  // format "<name>:<type>" or "<name>:<type>=<default>"
  // where <name> matches regexp [a-zA-Z][a-zA-Z0-9_]*
  // (by convention only using capital letters for attrs that can be inferred)
  // <type> can be:
  //   "string", "int", "float", "bool", "type", "shape", or "tensor"
  //   "numbertype", "realnumbertype", "quantizedtype"
  //       (meaning "type" with a restriction on valid values)
  //   "{int32,int64}" or {realnumbertype,quantizedtype,string}"
  //       (meaning "type" with a restriction containing unions of value types)
  //   "{\"foo\", \"bar\n baz\"}", or "{'foo', 'bar\n baz'}"
  //       (meaning "string" with a restriction on valid values)
  //   "list(string)", ..., "list(tensor)", "list(numbertype)", ...
  //       (meaning lists of the above types)
  //   "int >= 2" (meaning "int" with a restriction on valid values)
  //   "list(string) >= 2", "list(int) >= 2"
  //       (meaning "list(string)" / "list(int)" with length at least 2)
  // <default>, if included, should use the Proto text format
  // of <type>.  For lists use [a, b, c] format.
  //
  // Note that any attr specifying the length of an input or output will
  // get a default minimum of 1 unless the >= # syntax is used.
  //
  // TODO(josh11b): Perhaps support restrictions and defaults as optional
  // extra arguments to Attr() instead of encoding them in the spec string.
  // TODO(josh11b): Would like to have better dtype handling for tensor attrs:
  // * Ability to say the type of an input/output matches the type of
  //   the tensor.
  // * Ability to restrict the type of the tensor like the existing
  //   restrictions for type attrs.
  // Perhaps by linking the type of the tensor to a type attr?
  OpDefBuilder& Attr(string spec);

  // Adds an input or output to this OpDefBuilder (and returns *this).
  // The spec has form "<name>:<type-expr>" or "<name>:Ref(<type-expr>)"
  // where <name> matches regexp [a-z][a-z0-9_]* and <type-expr> can be:
  // * For a single tensor: <type>
  // * For a sequence of tensors with the same type: <number>*<type>
  // * For a sequence of tensors with different types: <type-list>
  // Where:
  //   <type> is either one of "float", "int32", "string", ...
  //                 or the name of an attr (see above) with type "type".
  //   <number> is the name of an attr with type "int".
  //   <type-list> is the name of an attr with type "list(type)".
  // TODO(josh11b): Indicate Ref() via an optional argument instead of
  // in the spec?
  // TODO(josh11b): SparseInput() and SparseOutput() matching the Python
  // handling?
  OpDefBuilder& Input(string spec);
  OpDefBuilder& Output(string spec);

  // Turns on the indicated boolean flag in this OpDefBuilder (and
  // returns *this).
  OpDefBuilder& SetIsCommutative();
  OpDefBuilder& SetIsAggregate();
  OpDefBuilder& SetIsStateful();
  OpDefBuilder& SetAllowsUninitializedInput();

  // Deprecate the op at a certain GraphDef version.
  OpDefBuilder& Deprecated(int version, string explanation);

  // Adds docs to this OpDefBuilder (and returns *this).
  // Docs have the format:
  //   <1-line summary>
  //   <rest of the description>
  //   <name>: <description of name>
  //   <name>: <description of name>
  //     <if long, indent the description on subsequent lines>
  // Where <name> is the name of an attr, input, or output.  Please
  // wrap docs at 72 columns so that it may be indented in the
  // generated output.  For tensor inputs or outputs (not attrs), you
  // may start the description with an "=" (like name:= <description>)
  // to suppress the automatically-generated type documentation in
  // generated output.
#ifndef TF_LEAN_BINARY
  OpDefBuilder& Doc(string text);
#else
  OpDefBuilder& Doc(string text) { return *this; }
#endif

  // Sets the shape function to be used for shape inference.
  //
  // Note that currently (October 2016), python code still requires a
  // RegisterShape call to invoke this; see call_cpp_shape_fn in
  // python/framework/common_shapes.py
  OpDefBuilder& SetShapeFn(OpShapeInferenceFn fn);

  // Sets op_reg_data->op_def to the requested OpDef and
  // op_reg_data->shape_inference_fn to the requested shape inference function,
  // or returns an error.
  // Must be called after all of the above methods.
  //
  // Note that OpDefBuilder only reports parsing errors.  You should also
  // call ValidateOpDef() to detect other problems.
  Status Finalize(OpRegistrationData* op_reg_data) const;

 private:
  friend class FunctionDefHelper;

  // Adds control output to this OpDefBuilder (and returns *this).
  // The <name> must be a valid node name (matches regexp
  // [a-zA-Z][a-zA-Z0-9_]*). Named control output can only exist for functions.
  OpDefBuilder& ControlOutput(string name);

  OpDef* op_def() { return &op_reg_data_.op_def; }

  OpRegistrationData op_reg_data_;
  std::vector<string> attrs_;
  std::vector<string> inputs_;
  std::vector<string> outputs_;
  std::vector<string> control_outputs_;
  string doc_;
  std::vector<string> errors_;
};

```

对类的定义进行下简化：

```c++
// Builder class passed to the REGISTER_OP() macro.
class OpDefBuilder {
 public:
  // Constructs an OpDef with just the name field set.
  explicit OpDefBuilder(string op_name);
  OpDefBuilder& Attr(string spec);
  OpDefBuilder& Input(string spec);
  OpDefBuilder& Output(string spec);
  OpDefBuilder& SetIsCommutative();
  OpDefBuilder& SetIsAggregate();
  OpDefBuilder& SetIsStateful();
  OpDefBuilder& SetAllowsUninitializedInput();
  OpDefBuilder& Deprecated(int version, string explanation);
  OpDefBuilder& Doc(string text);
  OpDefBuilder& SetShapeFn(OpShapeInferenceFn fn);
  Status Finalize(OpRegistrationData* op_reg_data) const;

 private:
  friend class FunctionDefHelper;
  OpDefBuilder& ControlOutput(string name);
  OpDef* op_def() { return &op_reg_data_.op_def; }
  OpRegistrationData op_reg_data_;
  std::vector<string> attrs_;
  std::vector<string> inputs_;
  std::vector<string> outputs_;
  std::vector<string> control_outputs_;
  string doc_;
  std::vector<string> errors_;
};
```

上面除了构造函数需要传入 `op_name` 外，只有 `Input, Output, Attr, SetShapeFn, Doc, Deprecated` 需要我们传入参数。Doc 的功能顾名思义，Input, Output, Attr 需要我们按 `spec` 规则来传入符合规范的字符串，而 `SetShapeFn` 需要我们传入一个回调函数。

- `Attr(string spec);` 为 `OpDefBuilder` [添加属性](https://www.tensorflow.org/guide/extend/op#op_registration)。

  spec 的格式为："<name>:<type>" 或 "<name>:<type>=<default>"

  - <name>： 满足 `[a-zA-Z][a-zA-Z0-9_]*` 正则
  - <type>：可以是一个简单的基础类型，也可以是复合类型：
    - 基础类型：`string, int, float, bool, type, shape 或 tensor`等。
    - 复合类型：`{int32,int64}, numbertype, realnumbertype ` 等。

  - <default>： 为该属性添加默认值。

- `Input(string spec)` 

- `Output(string sepc)`

  `Input, Output` 定义的 spec 规范是一样的。

  spec 的格式为："<name>:<type-expr>" 或 "<name>:Ref(<type-expr>)"

  - <name> :  满足 `[a-zA-Z][a-zA-Z0-9_]*` 正则

  - <type-expr>: 可以是：

    - 一个 tensor 的 type: `<type>`。即我们在操作张量时的 dtype.

      ```python
      In [106]: x = tf.matmul([[1]], [[2, 3]])
      
      In [107]: print(x)
      tf.Tensor([[2 3]], shape=(1, 2), dtype=int32)
      
      In [100]: x = tf.sin(1.0)
      
      In [101]: print(x)
      tf.Tensor(0.84147096, shape=(), dtype=float32)
      
      In [98]: type(x)
      Out[98]: tensorflow.python.framework.ops.EagerTensor
      
      In [99]: print(x)
      tf.Tensor(
      [[0.35174477 0.88608444 0.72226477]
       [0.21097851 0.12061656 0.81054926]
       [0.7454673  0.06144464 0.46765244]], shape=(3, 3), dtype=float32)
      ```

      我们只需要定义，Tensor 的 `dtype` 就可以了，不要关心什么 `shape`. 

    - 同一类型的 tensor: `<number> * <type>`

    - 不同类型的 tensor list: `[type-list]`

  其中：`<type>, <number>, [type-list]` 的定义如下：

  - <type> 可以是 `float, int32` 基础类型，也是复合类型。和 attr 中的 type 一样。
  - <number> 是定义为 `int` 属性的属性名。
  - <type-list> 是定义为 `list(type)` 属性的属性名。

  ```c++
    // Adds an input or output to this OpDefBuilder (and returns *this).
    // The spec has form "<name>:<type-expr>" or "<name>:Ref(<type-expr>)"
    // where <name> matches regexp [a-z][a-z0-9_]* and <type-expr> can be:
    // * For a single tensor: <type>
    // * For a sequence of tensors with the same type: <number>*<type>
    // * For a sequence of tensors with different types: <type-list>
    // Where:
    //   <type> is either one of "float", "int32", "string", ...
    //                 or the name of an attr (see above) with type "type".
    //   <number> is the name of an attr with type "int".
    //   <type-list> is the name of an attr with type "list(type)".
  ```

  当然 ，当我们需要添加的操作即需要支持 int 又要支持 float 怎么办那？我们当然也可以使用泛型来编程了，即, [多态类型](https://www.tensorflow.org/guide/extend/op#type_polymorphism)：

  ```c++
  REGISTER_OP("ZeroOut")
    .Attr("T: {float, int32} = DT_INT32")
    .Input("to_zero: T")
    .Output("zeroed: T")
  ```

我们需要为每一种类型注册一个 `OpKernel`. 上面的操作注册说明输入类型必须是 `float` 和 `int32` 的，并且输出也是同一个类型，因为它们都具有 `T` 类型。

那么我们可能需要写义两个 `Opkernel` ， 当然用泛型可以不用定义两个，但注册时还是需要的。即如下：

```c++
// Note that TypeConstraint<int32>("T") means that attr "T" (defined
// in the op registration above) must be "int32" to use this template
// instantiation.
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("T"),
    ZeroOutOpInt32);

REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    ZeroOutFloatOp);
```

在 `Input, Output` 中我们只定义了张量的类型，但是并没定义张量的形状，如果要定义张量的形状该怎么办那？要对张量的形状做验证和定义那就需要用到 InferenceContext 了。我们在例子中已经看到了 `SetShapeFn` 的用法。即：

```c++
REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

基中 c 是 InferenceContext 对象。InferenceContext 定义了些常用的方法，如：

- `ShapeHandle input(int64 idx)` ：根据下标访问 Input 的值。其源代码如下：

  ```c++
  ShapeHandle input(int64 idx) const { return inputs_[idx]; }
  ```

  那么我们不禁要问 `inputs_` 是那里定义的，在那里赋值的？

  这时的 `inputs_` 其实是基于我们的在定义操作时生成的，在注册时调用 `.Input("zeroed: int32")` 方法时，其源码是这样的：

  ```c++
  OpDefBuilder& OpDefBuilder::Input(string spec) {
    inputs_.push_back(std::move(spec));
    return *this;
  }
  ```

  所在 `SetShapeFn` 中 c-input(0) 获取第一个 (下标为 0 ) 的输入 `ShapeHandle` 就是我定义时调用的第一个 Input. 可见 Input, Output 我们可以定义多个，在验证时根据下标来验证，设置输出也根据下标输出。

- set_output(index, shape)`： 根据下标设置 Out 的值。

#### InferenceContext 

```c++
// Shape inference functions registered on ops in REGISTER_OP implement
// their shape functions in terms of this InferenceContext.  An InferenceContext
// is created by the framework and passed to a shape inference function.  The
// shape inference function calls functions on the context, and should call
// set_output() to set the shape on all outputs.
//
// To infer shapes for user-defined functions see ShapeRefiner.
//
// All Shape* and Dimension* returned by functions of InferenceContext are owned
// by the InferenceContext.
class InferenceContext {
 public:
  static constexpr int64 kUnknownDim = -1;
  static constexpr int32 kUnknownRank = -1;

  // <input_tensors> is NULL-padded to be the same size as <input_shapes>.
  //
  // Elements of <input_tensors_as_shapes> are used for when a shape function
  // makes a call to MakeShapeFromShapeTensor; in particular, when the
  // input_tensors[i] is nullptr but the shape represented by it is partially
  // known from analysis of the graph.
  // <input_tensors_as_shapes> can have fewer elements than <input_shapes>.
  // Values of <input_tensors_as_shapes> do not need to outlive the context.
  //
  // REQUIRES: <node_def> is not NULL, and must outlive the InferenceContext.
  InferenceContext(int graph_def_version, const NodeDef* node_def,
                   const OpDef& op_def,
                   const std::vector<ShapeHandle>& input_shapes,
                   const std::vector<const Tensor*>& input_tensors,
                   const std::vector<ShapeHandle>& input_tensors_as_shapes,
                   std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
                       input_handle_shapes_and_types);

  // <input_tensors> is NULL-padded to be the same size as <input_shapes>.
  //
  // Elements of <input_tensors_as_shapes> are used for when a shape
  // function makes a call to MakeShapeFromShapeTensor; in particular, when
  // the input_tensors[i] is nullptr but the shape represented by it is
  // partially known from analysis of the graph. <input_tensors_as_shapes>
  // can have fewer elements than <input_shapes>. Values of
  // <input_tensors_as_shapes> do not need to outlive the context.
  //
  // REQUIRES: <node_def> is not NULL, and must outlive the
  // InferenceContext.
  InferenceContext(
      int graph_def_version, const NodeDef* node_def, const OpDef& op_def,
      const std::vector<TensorShapeProto>& input_shapes,
      const std::vector<const Tensor*>& input_tensors,
      const std::vector<TensorShapeProto>& input_tensors_as_shapes,
      const std::vector<
          std::unique_ptr<std::vector<std::pair<TensorShapeProto, DataType>>>>&
          input_handle_shapes_and_types);

  InferenceContext(
      int graph_def_version, const NodeDef* node_def, const OpDef& op_def,
      const std::vector<PartialTensorShape>& input_shapes,
      const std::vector<const Tensor*>& input_tensors,
      const std::vector<PartialTensorShape>& input_tensors_as_shapes,
      const std::vector<std::unique_ptr<
          std::vector<std::pair<PartialTensorShape, DataType>>>>&
          input_handle_shapes_and_types);

  ~InferenceContext();

  Status Run(
      const std::function<Status(shape_inference::InferenceContext* c)>& fn);

  bool MergeInput(int idx, ShapeHandle shape);

  bool RelaxInput(int idx, ShapeHandle shape);

  void SetInput(int idx, ShapeHandle shape) { inputs_[idx] = shape; }

  ShapeHandle input(int64 idx) const { return inputs_[idx]; }
  Status input(StringPiece input_name, std::vector<ShapeHandle>* output) const;
  int num_inputs() const { return inputs_.size(); }
  const Tensor* input_tensor(int idx);
  bool requested_input_tensor(int idx) const;
  bool requested_input_tensor_as_partial_shape(int idx) const;

  void set_input_tensors(const std::vector<const Tensor*>& input_tensors);

  void set_input_tensors_as_shapes(
      const std::vector<ShapeHandle>& input_tensors_as_shapes);

  const std::vector<ShapeHandle>& input_tensors_as_shapes() const;

  ShapeHandle output(int64 idx) const { return outputs_.at(idx); }
  void set_output(int idx, ShapeHandle shape) { outputs_.at(idx) = shape; }
  Status set_output(StringPiece output_name,
                    const std::vector<ShapeHandle>& shapes);

  int num_outputs() const { return outputs_.size(); }
  ShapeHandle output(int idx) const { return outputs_.at(idx); }
  Status output(StringPiece output_name,
                std::vector<ShapeHandle>* output) const;

  AttrSlice attrs() const { return AttrSlice(*node_def_); }

  string op() const;

  // idx can be negative for an offset from end of dimensions.
  // idx must be in the range [-1 * s.rank, s.rank).
  DimensionHandle Dim(ShapeHandle s, int64 idx);
  // As above, but asserts that the rank of the shape is known.
  static DimensionHandle DimKnownRank(ShapeHandle s, int64 idx);

  static int32 Rank(ShapeHandle s) {
    DCHECK(s.IsSet());
    return s.IsSet() ? s->rank_ : kUnknownRank;
  }
  static bool RankKnown(ShapeHandle s) {
    return (s.IsSet() && (Rank(s) != kUnknownRank));
  }
  static inline int64 Value(DimensionOrConstant d) {
    return d.dim.IsSet() ? d.dim->value_ : d.val;
  }
  static inline bool ValueKnown(DimensionOrConstant d) {
    return Value(d) != kUnknownDim;
  }

  // Fills the output proto with the shape defined by the handle.
  // "proto" is expected to be empty prior to the call.
  void ShapeHandleToProto(ShapeHandle handle, TensorShapeProto* proto);

  // Returns true if the rank and all dimensions of the Shape are known.
  bool FullyDefined(ShapeHandle s);

  // Returns the total number of elements, or an unknown dimension for an
  // incomplete shape.
  DimensionHandle NumElements(ShapeHandle s);

  // If <shape> has rank <rank>, or its rank is unknown, return OK and return
  // the shape with asserted rank in <*out>. Otherwise return an error.
  //
  // Note that <*out> may be set to <shape>.
  Status WithRank(ShapeHandle shape, int64 rank,
                  ShapeHandle* out) TF_MUST_USE_RESULT;
  Status WithRankAtLeast(ShapeHandle shape, int64 rank,
                         ShapeHandle* out) TF_MUST_USE_RESULT;
  Status WithRankAtMost(ShapeHandle shape, int64 rank,
                        ShapeHandle* out) TF_MUST_USE_RESULT;

  // If <dim> has value <value>, or its value is unknown, returns OK and returns
  // the dimension with asserted value in <*out>. Otherwise returns an error.
  //
  // Note that <*out> may be set to <dim>.
  Status WithValue(DimensionHandle dim, int64 value,
                   DimensionHandle* out) TF_MUST_USE_RESULT;

  // Merges <s0> and <s1> and returns the merged shape in <*out>. See
  // 'MergeInput' function for full details and examples.
  Status Merge(ShapeHandle s0, ShapeHandle s1,
               ShapeHandle* out) TF_MUST_USE_RESULT;

  // Asserts that <s>'s rank >= <prefix>'s rank, and the first
  // <prefix.rank> dimensions of <s> are compatible with the dimensions of
  // <prefix>.
  // Returns the merged results in <*s_out> and <*prefix_out>.
  Status MergePrefix(ShapeHandle s, ShapeHandle prefix, ShapeHandle* s_out,
                     ShapeHandle* prefix_out) TF_MUST_USE_RESULT;

  // Merges <d0> and <d1> and returns the merged dimension in <*out>. If <d0>
  // and <d1> have incompatible values, returns an error.
  //
  // Note that <*out> may be set to <d0> or <d1>.
  Status Merge(DimensionHandle d0, DimensionHandle d1,
               DimensionHandle* out) TF_MUST_USE_RESULT;

  // Returns in <*out> a sub-shape of <s> with dimensions [start:].
  // <start> can be negative to index from the end of the shape. If <start> >
  // rank of <s>, then an empty subshape is returned.
  Status Subshape(ShapeHandle s, int64 start,
                  ShapeHandle* out) TF_MUST_USE_RESULT;

  // Returns in <*out> a sub-shape of <s>, with dimensions [start:end].
  // <start> and <end> can be negative, to index from the end of the shape.
  // <start> and <end> are set to the rank of <s> if > rank of <s>.
  Status Subshape(ShapeHandle s, int64 start, int64 end,
                  ShapeHandle* out) TF_MUST_USE_RESULT;

  // Returns in <*out> a sub-shape of <s>, with dimensions [start:end:stride].
  // <start> and <end> can be negative, to index from the end of the shape.
  // <start> and <end> are set to the rank of <s> if > rank of <s>.
  // <stride> can be negative, to reverse the <s>.
  Status Subshape(ShapeHandle s, int64 start, int64 end, int64 stride,
                  ShapeHandle* out) TF_MUST_USE_RESULT;

  // Returns in <*out> the result of appending the dimensions of <s2> to those
  // of <s1>.
  Status Concatenate(ShapeHandle s1, ShapeHandle s2,
                     ShapeHandle* out) TF_MUST_USE_RESULT;

  // Returns in <out> the shape from replacing <s.dim[dim_index]> with
  // <new_dim>.
  Status ReplaceDim(ShapeHandle s, int64 dim_index, DimensionHandle new_dim,
                    ShapeHandle* out) TF_MUST_USE_RESULT;

  // Returns a new shape with the given dims. The returned value is owned by
  // this context.
  ShapeHandle MakeShape(const std::vector<DimensionHandle>& dims);
  ShapeHandle MakeShape(std::initializer_list<DimensionOrConstant> dims);

  // Returns a new unknown shape.
  ShapeHandle UnknownShape();

  // Returns a shape with specified rank but unknown dims.
  ShapeHandle UnknownShapeOfRank(int64 rank);

  // Returns a new shape of zero dimensions.
  ShapeHandle Scalar();

  // Returns a new shape of one dimension.
  ShapeHandle Vector(DimensionOrConstant dim);

  // Returns a new shape of two dimensions.
  ShapeHandle Matrix(DimensionOrConstant dim1, DimensionOrConstant dim2);

  // Returns in <out> a new shape whose dimension sizes come from input tensor
  // <input_idx>. The tensor must be a 1-dimensional int32 or int64 tensor.  If
  // the input tensor is NULL, then an unknown shape is returned.
  Status MakeShapeFromShapeTensor(int input_idx, ShapeHandle* out);

  // Like the function above, but treats scalar values as unknown
  // shapes.  **NOTE** If the scalar is statically known, its value
  // must be -1 or an error is returned.
  Status MakeShapeFromShapeTensorTreatScalarAsUnknownShape(int input_idx,
                                                           ShapeHandle* out);

  // Returns in <out> a new shape corresponding to <proto>.
  Status MakeShapeFromShapeProto(const TensorShapeProto& proto,
                                 ShapeHandle* out);

  // Returns in <out> a new shape corresponding to <partial_shape>.
  Status MakeShapeFromPartialTensorShape(
      const PartialTensorShape& partial_shape, ShapeHandle* out);

  // Returns in <out> a new shape corresponding to <shape>.
  Status MakeShapeFromTensorShape(const TensorShape& shape, ShapeHandle* out);

  // Returns a new dimension of the given size.  The returned value is owned by
  // this context.
  inline DimensionHandle MakeDim(DimensionOrConstant d) {
    return shape_manager_.MakeDim(d);
  }

  inline DimensionHandle UnknownDim() { return MakeDim(kUnknownDim); }

  // Returns in <val> a scalar value from an input tensor <t>.  The input tensor
  // must be a 1-dimensional int32 or int64 tensor.  Caller must ensure that the
  // input tensor is not NULL.
  Status GetScalarFromTensor(const Tensor* t, int64* val);

  // Returns a new dimension whose value is given by a scalar input tensor.
  // The input tensor must be in host memory, since it is dereferenced to get
  // the value.
  Status MakeDimForScalarInput(int idx, DimensionHandle* out);

  // Returns a new dimension whose value is given by a scalar input tensor.
  // This allows for a negative input dimension given the rank of a separate
  // tensor.  This rank can be negative if unknown.
  // The input tensor must be in host memory, since it is dereferenced to get
  // the value.
  Status MakeDimForScalarInputWithNegativeIndexing(int idx, int input_rank,
                                                   DimensionHandle* out);

  // Look up the attr for the NodeDef being evaluated with name attr_name and
  // set *value to its value.  If no attr with attr_name is found in def(), or
  // the attr does not have a matching type, a non-ok status will be returned.
  template <class T>
  Status GetAttr(StringPiece attr_name, T* value) const;

  // Returns in <out> the result of dividing <dividend> by <divisor>.
  // Returns an error if <divisor>  is not positive or if <evenly_divisible>
  // and <divisor> does not evenly divide <dividend>.
  Status Divide(DimensionHandle dividend, DimensionOrConstant divisor,
                bool evenly_divisible, DimensionHandle* out);

  // Returns in <out> the sum of <first> and <second>.
  Status Add(DimensionHandle first, DimensionOrConstant second,
             DimensionHandle* out);

  // Returns in <out> the dimension that is <first> minus <second>.
  Status Subtract(DimensionHandle first, DimensionOrConstant second,
                  DimensionHandle* out);

  // Returns in <out> the product of <first> and <second>.
  Status Multiply(DimensionHandle first, DimensionOrConstant second,
                  DimensionHandle* out);

  // Returns in <out> the minimum of <first> and <second>. If either <first> or
  // <second> is zero the results is zero. Otherwise, if either <first> or
  // <second> is unknown the results is unknown.
  Status Min(DimensionHandle first, DimensionOrConstant second,
             DimensionHandle* out);

  // Returns in <out> the maximum of <first> and <second>. If either <first> or
  // <second> is unknown the results is unknown.
  Status Max(DimensionHandle first, DimensionOrConstant second,
             DimensionHandle* out);

  Status construction_status() const { return construction_status_; }

  bool MergeInputHandleShapesAndTypes(
      int idx,
      const std::vector<ShapeAndType>& shapes_and_types) TF_MUST_USE_RESULT;

  // As MergeInputHandleShapesAndTypes, but for an output.
  bool MergeOutputHandleShapesAndTypes(
      int idx,
      const std::vector<ShapeAndType>& shapes_and_types) TF_MUST_USE_RESULT;

  bool RelaxInputHandleShapesAndMergeTypes(
      int idx,
      const std::vector<ShapeAndType>& shapes_and_types) TF_MUST_USE_RESULT;

  // As RelaxInputHandleShapesAndTypes, but for an output.
  bool RelaxOutputHandleShapesAndMergeTypes(
      int idx,
      const std::vector<ShapeAndType>& shapes_and_types) TF_MUST_USE_RESULT;

  void set_input_handle_shapes_and_types(
      int idx, const std::vector<ShapeAndType>& shapes_and_types) {
    input_handle_shapes_and_types_[idx].reset(
        new std::vector<ShapeAndType>(shapes_and_types));
  }

  // Returns the output handle shapes and types, for the resource tensor output
  // at index <idx>. Returns NULL if the shape and types were never set.
  const std::vector<ShapeAndType>* output_handle_shapes_and_types(int idx) {
    return output_handle_shapes_and_types_[idx].get();
  }

  // Returns the inputs handle shapes and types, for the resource tensor output
  // at index <idx>. Returns NULL if the shape and types were not available.
  const std::vector<ShapeAndType>* input_handle_shapes_and_types(int idx) {
    return input_handle_shapes_and_types_[idx].get();
  }

  void set_output_handle_shapes_and_types(
      int idx, const std::vector<ShapeAndType>& shapes_and_types) {
    output_handle_shapes_and_types_[idx].reset(
        new std::vector<ShapeAndType>(shapes_and_types));
  }

  // Note that shape functions should usually call MakeShapeFromShapeTensor,
  // as it does more analysis to provide partial shapes.
  //
  // Returns in <out> a new shape whose dimension sizes come from tensor <t>.
  // The tensor must be a 1-dimensional int32 or int64 tensor.  If <t> is NULL,
  // then an unknown shape is returned.
  Status MakeShapeFromTensor(const Tensor* t, ShapeHandle tensor_shape,
                             ShapeHandle* out);

  int graph_def_version() const { return graph_def_version_; }

  const std::vector<std::pair<ShapeHandle, ShapeHandle>>& MergedShapes() const {
    return merged_shapes_;
  }
  const std::vector<std::pair<DimensionHandle, DimensionHandle>>& MergedDims()
      const {
    return merged_dims_;
  }

  // Adds new outputs; useful when mutating the graph.
  Status ExpandOutputs(int new_output_size);

 private:
  // Creates and stores shapes for use in InferenceContext.
  class ShapeManager {
   public:
    ShapeManager();
    ~ShapeManager();

    // Returns a new shape with the given dims. The returned value is owned by
    // this class.
    ShapeHandle MakeShape(const std::vector<DimensionHandle>& dims);

    // Returns a new unknown shape.
    ShapeHandle UnknownShape();

    // Returns a new dimension of the given size.  The returned value
    // is owned by this class.
    inline DimensionHandle MakeDim(DimensionOrConstant d) {
      if (d.dim.IsSet()) {
        return d.dim;
      } else {
        all_dims_.push_back(new Dimension(d.val));
        return all_dims_.back();
      }
    }

   private:
    std::vector<Shape*> all_shapes_;    // values are owned.
    std::vector<Dimension*> all_dims_;  // values are owned.
  };

  friend class ::tensorflow::grappler::GraphProperties;

  // Friend for user-defined function shape inference purposes.
  friend class ::tensorflow::ShapeRefiner;

  friend class ShapeInferenceTest;      // For testing Relax functions.
  friend class ShapeInferenceTestutil;  // For testing shapes.

  // Shared initialization across the two constructors.  Remove
  // once we get rid of one of them.
  void PreInputInit(const OpDef& op_def,
                    const std::vector<const Tensor*>& input_tensors,
                    const std::vector<ShapeHandle>& input_tensors_as_shapes);
  void PostInputInit(std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
                         input_handle_data);

  DimensionHandle GetDimension(const DimensionOrConstant& d);

  Status ReturnUnknownShape(ShapeHandle* out) {
    *out = UnknownShape();
    return Status::OK();
  }
  Status ReturnCreatedShape(const std::vector<DimensionHandle>& dims,
                            ShapeHandle* out) {
    *out = MakeShape(dims);
    return Status::OK();
  }

  // Adds additional context to the given status.
  Status AttachContext(const Status& status);

  // Relaxes an existing value <d_old> with a new value <d_new> and returns the
  // relaxed dimension in <*out>. If <d_old> and <d_new> have incompatible
  // values, returns an error.
  //
  // Note that <*out> may be set to <d_old> or <d_new>.
  void Relax(DimensionHandle d_old, DimensionHandle d_new,
             DimensionHandle* out);
  // Relaxes an existing shape <s_old> with a new shape <s_new> and returns the
  // relaxed shape in <*out>. See 'RelaxInput' function for full details and
  // examples.
  void Relax(ShapeHandle s_old, ShapeHandle s_new, ShapeHandle* out);

  // Used to implement MergeInputHandleShapesAndTypes and
  // MergeOutputHandleShapesAndTypes.
  bool MergeHandleShapesAndTypes(
      const std::vector<ShapeAndType>& shapes_and_types,
      std::vector<ShapeAndType>* to_update) TF_MUST_USE_RESULT;
  // Used to implement RelaxInputHandleShapesAndMergeTypes and
  // RelaxOutputHandleShapesAndMergeTypes.
  bool RelaxHandleShapesAndMergeTypes(
      const std::vector<ShapeAndType>& shapes_and_types,
      std::vector<ShapeAndType>* to_update) TF_MUST_USE_RESULT;

  // Forget all the previous merged shapes and dims.
  void ForgetMerges() {
    merged_shapes_.clear();
    merged_dims_.clear();
  }

  // Helper method for MakeShapeFromTensor and MakeShapeFromShapeTensor.
  Status InternalMakeShapeFromTensor(
      bool treat_unknown_scalar_tensor_as_unknown_shape, const Tensor* t,
      ShapeHandle tensor_shape, ShapeHandle* out);

  ShapeManager shape_manager_;

  // inputs_, outputs_, and input_tensors_as_shapes_ refer to values from
  // `shape_manager_`.
  std::vector<ShapeHandle> inputs_;
  std::vector<const Tensor*> input_tensors_;
  std::vector<bool> requested_input_tensor_;
  std::vector<ShapeHandle> outputs_;
  // Can have fewer elements than inputs_.
  std::vector<ShapeHandle> input_tensors_as_shapes_;
  std::vector<bool> requested_input_tensor_as_partial_shape_;

  // input_handle_shapes_and_types_[i] is the list of shape/type pairs available
  // through the resource handle passed along input i of the node.
  //
  // Values may be NULL.
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
      input_handle_shapes_and_types_;

  // output_handle_shapes_and_types_[i] is the list of shape/type pairs
  // available through the resource handle passed along output i of the node.
  //
  // Values may be NULL.
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
      output_handle_shapes_and_types_;

  const int graph_def_version_;
  const NodeDef* node_def_;
  NameRangeMap input_name_map_;
  NameRangeMap output_name_map_;

  // An error set during construction. TODO(cwhipkey): remove when test
  // constructor is removed.
  Status construction_status_;

  // Pair of shape or dim handles that are equivalent, ie that represent the
  // same underlying shape of dimension. Note that for each pair at least one of
  // the handles must contain an unknown shape, since we don't keep track of
  // known shapes or dims here.
  std::vector<std::pair<ShapeHandle, ShapeHandle>> merged_shapes_;
  std::vector<std::pair<DimensionHandle, DimensionHandle>> merged_dims_;

  TF_DISALLOW_COPY_AND_ASSIGN(InferenceContext);
};
```

具体的 Rank ， Status 检查等操作直接看这个文件的 API 即可。如常用的：

- `WithRank` , 检查 Rank 并且返回 shape.

  ```c++
    // If <shape> has rank <rank>, or its rank is unknown, return OK and return
    // the shape with asserted rank in <*out>. Otherwise return an error.
    //
    // Note that <*out> may be set to <shape>.
    Status WithRank(ShapeHandle shape, int64 rank,
                    ShapeHandle* out) TF_MUST_USE_RESULT;
  ```

- `Dim`, 返回相应维度的  Dim

  ```c++
  // idx can be negative for an offset from end of dimensions.
  // idx must be in the range [-1 * s.rank, s.rank).
  DimensionHandle Dim(ShapeHandle s, int64 idx) {
    if (s->rank_ == kUnknownRank) {
      return UnknownDim();
    }
    return DimKnownRank(s, idx);
  }
  ```

- `Merge`

  ```c++
    // Merges <s0> and <s1> and returns the merged shape in <*out>. See
    // 'MergeInput' function for full details and examples.
    Status Merge(ShapeHandle s0, ShapeHandle s1,
                 ShapeHandle* out) TF_MUST_USE_RESULT;
  ```

- `Matrix`

  ```c++
  // Returns a new shape of two dimensions.
  ShapeHandle Matrix(DimensionOrConstant dim1, DimensionOrConstant dim2);
  ```

到目前为止终于把注册这一步给完成了。