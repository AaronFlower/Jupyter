џН
Ћ§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8уА
|
Conv1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_nameConv1/kernel
u
 Conv1/kernel/Read/ReadVariableOpReadVariableOpConv1/kernel*
dtype0*&
_output_shapes
:
l

Conv1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_name
Conv1/bias
e
Conv1/bias/Read/ReadVariableOpReadVariableOp
Conv1/bias*
dtype0*
_output_shapes
:
y
Softmax/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	Ш

*
shared_nameSoftmax/kernel
r
"Softmax/kernel/Read/ReadVariableOpReadVariableOpSoftmax/kernel*
dtype0*
_output_shapes
:	Ш


p
Softmax/biasVarHandleOp*
shape:
*
shared_nameSoftmax/bias*
dtype0*
_output_shapes
: 
i
 Softmax/bias/Read/ReadVariableOpReadVariableOpSoftmax/bias*
dtype0*
_output_shapes
:

f
	Adam/iterVarHandleOp*
shared_name	Adam/iter*
dtype0	*
_output_shapes
: *
shape: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shape: *
shared_name
Adam/decay*
dtype0*
_output_shapes
: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: *
shape: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shape: *
shared_nametotal*
dtype0*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 

Adam/Conv1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*$
shared_nameAdam/Conv1/kernel/m

'Adam/Conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1/kernel/m*
dtype0*&
_output_shapes
:
z
Adam/Conv1/bias/mVarHandleOp*
shape:*"
shared_nameAdam/Conv1/bias/m*
dtype0*
_output_shapes
: 
s
%Adam/Conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1/bias/m*
dtype0*
_output_shapes
:

Adam/Softmax/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:	Ш

*&
shared_nameAdam/Softmax/kernel/m

)Adam/Softmax/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Softmax/kernel/m*
dtype0*
_output_shapes
:	Ш


~
Adam/Softmax/bias/mVarHandleOp*$
shared_nameAdam/Softmax/bias/m*
dtype0*
_output_shapes
: *
shape:

w
'Adam/Softmax/bias/m/Read/ReadVariableOpReadVariableOpAdam/Softmax/bias/m*
dtype0*
_output_shapes
:


Adam/Conv1/kernel/vVarHandleOp*$
shared_nameAdam/Conv1/kernel/v*
dtype0*
_output_shapes
: *
shape:

'Adam/Conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1/kernel/v*
dtype0*&
_output_shapes
:
z
Adam/Conv1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*"
shared_nameAdam/Conv1/bias/v
s
%Adam/Conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1/bias/v*
dtype0*
_output_shapes
:

Adam/Softmax/kernel/vVarHandleOp*
shape:	Ш

*&
shared_nameAdam/Softmax/kernel/v*
dtype0*
_output_shapes
: 

)Adam/Softmax/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Softmax/kernel/v*
dtype0*
_output_shapes
:	Ш


~
Adam/Softmax/bias/vVarHandleOp*
shape:
*$
shared_nameAdam/Softmax/bias/v*
dtype0*
_output_shapes
: 
w
'Adam/Softmax/bias/v/Read/ReadVariableOpReadVariableOpAdam/Softmax/bias/v*
dtype0*
_output_shapes
:


NoOpNoOp
и
ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB Bџ
й
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

iter

 beta_1

!beta_2
	"decay
#learning_ratemDmEmFmGvHvIvJvK

0
1
2
3

0
1
2
3
 


$layers
trainable_variables
	variables
%non_trainable_variables
&metrics
'layer_regularization_losses
regularization_losses
 
 
 
 


(layers
trainable_variables
	variables
)non_trainable_variables
*metrics
+layer_regularization_losses
regularization_losses
XV
VARIABLE_VALUEConv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
Conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 


,layers
trainable_variables
	variables
-non_trainable_variables
.metrics
/layer_regularization_losses
regularization_losses
 
 
 


0layers
trainable_variables
	variables
1non_trainable_variables
2metrics
3layer_regularization_losses
regularization_losses
ZX
VARIABLE_VALUESoftmax/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUESoftmax/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 


4layers
trainable_variables
	variables
5non_trainable_variables
6metrics
7layer_regularization_losses
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

80
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	9total
	:count
;
_fn_kwargs
<trainable_variables
=	variables
>regularization_losses
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

90
:1
 


@layers
<trainable_variables
=	variables
Anon_trainable_variables
Bmetrics
Clayer_regularization_losses
>regularization_losses
 

90
:1
 
 
{y
VARIABLE_VALUEAdam/Conv1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Conv1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Softmax/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Softmax/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Conv1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Conv1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Softmax/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Softmax/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_Conv1_inputPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџ*$
shape:џџџџџџџџџ
л
StatefulPartitionedCallStatefulPartitionedCallserving_default_Conv1_inputConv1/kernel
Conv1/biasSoftmax/kernelSoftmax/bias*
Tin	
2*'
_output_shapes
:џџџџџџџџџ
*-
_gradient_op_typePartitionedCall-152876*-
f(R&
$__inference_signature_wrapper_152745*
Tout
2**
config_proto

GPU 

CPU2J 8
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
Ш
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename Conv1/kernel/Read/ReadVariableOpConv1/bias/Read/ReadVariableOp"Softmax/kernel/Read/ReadVariableOp Softmax/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/Conv1/kernel/m/Read/ReadVariableOp%Adam/Conv1/bias/m/Read/ReadVariableOp)Adam/Softmax/kernel/m/Read/ReadVariableOp'Adam/Softmax/bias/m/Read/ReadVariableOp'Adam/Conv1/kernel/v/Read/ReadVariableOp%Adam/Conv1/bias/v/Read/ReadVariableOp)Adam/Softmax/kernel/v/Read/ReadVariableOp'Adam/Softmax/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
_output_shapes
: *-
_gradient_op_typePartitionedCall-152917*(
f#R!
__inference__traced_save_152916*
Tout
2**
config_proto

GPU 

CPU2J 8
Ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv1/kernel
Conv1/biasSoftmax/kernelSoftmax/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/Conv1/kernel/mAdam/Conv1/bias/mAdam/Softmax/kernel/mAdam/Softmax/bias/mAdam/Conv1/kernel/vAdam/Conv1/bias/vAdam/Softmax/kernel/vAdam/Softmax/bias/v*
Tin
2*
_output_shapes
: *-
_gradient_op_typePartitionedCall-152987*+
f&R$
"__inference__traced_restore_152986*
Tout
2**
config_proto

GPU 

CPU2J 8кн
Ј
и
F__inference_sequential_layer_call_and_return_conditional_losses_152672
conv1_input(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2*
&softmax_statefulpartitionedcall_args_1*
&softmax_statefulpartitionedcall_args_2
identityЂConv1/StatefulPartitionedCallЂSoftmax/StatefulPartitionedCall
Conv1/StatefulPartitionedCallStatefulPartitionedCallconv1_input$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-152610*J
fERC
A__inference_Conv1_layer_call_and_return_conditional_losses_152604*
Tout
2**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ*
Tin
2Т
flatten/PartitionedCallPartitionedCall&Conv1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-152636*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_152630*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџШ

Softmax/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&softmax_statefulpartitionedcall_args_1&softmax_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-152660*L
fGRE
C__inference_Softmax_layer_call_and_return_conditional_losses_152654*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ
*
Tin
2В
IdentityIdentity(Softmax/StatefulPartitionedCall:output:0^Conv1/StatefulPartitionedCall ^Softmax/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::2B
Softmax/StatefulPartitionedCallSoftmax/StatefulPartitionedCall2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall:+ '
%
_user_specified_nameConv1_input: : : : 
њ
_
C__inference_flatten_layer_call_and_return_conditional_losses_152811

inputs
identity^
Reshape/shapeConst*
valueB"џџџџH  *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ
Y
IdentityIdentityReshape:output:0*(
_output_shapes
:џџџџџџџџџШ
*
T0"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
Ј
и
F__inference_sequential_layer_call_and_return_conditional_losses_152685
conv1_input(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2*
&softmax_statefulpartitionedcall_args_1*
&softmax_statefulpartitionedcall_args_2
identityЂConv1/StatefulPartitionedCallЂSoftmax/StatefulPartitionedCall
Conv1/StatefulPartitionedCallStatefulPartitionedCallconv1_input$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*J
fERC
A__inference_Conv1_layer_call_and_return_conditional_losses_152604*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-152610Т
flatten/PartitionedCallPartitionedCall&Conv1/StatefulPartitionedCall:output:0*(
_output_shapes
:џџџџџџџџџШ
*
Tin
2*-
_gradient_op_typePartitionedCall-152636*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_152630*
Tout
2**
config_proto

GPU 

CPU2J 8
Softmax/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&softmax_statefulpartitionedcall_args_1&softmax_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-152660*L
fGRE
C__inference_Softmax_layer_call_and_return_conditional_losses_152654*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ
*
Tin
2В
IdentityIdentity(Softmax/StatefulPartitionedCall:output:0^Conv1/StatefulPartitionedCall ^Softmax/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::2B
Softmax/StatefulPartitionedCallSoftmax/StatefulPartitionedCall2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall: :+ '
%
_user_specified_nameConv1_input: : : 
Ц-

__inference__traced_save_152916
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop-
)savev2_softmax_kernel_read_readvariableop+
'savev2_softmax_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop4
0savev2_adam_softmax_kernel_m_read_readvariableop2
.savev2_adam_softmax_bias_m_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop4
0savev2_adam_softmax_kernel_v_read_readvariableop2
.savev2_adam_softmax_bias_v_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_1329d6ac646640e08dc76a09c2e7db9a/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Б

SaveV2/tensor_namesConst"/device:CPU:0*к	
valueа	BЭ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
SaveV2/shape_and_slicesConst"/device:CPU:0*9
value0B.B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:ш
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop)savev2_softmax_kernel_read_readvariableop'savev2_softmax_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop0savev2_adam_softmax_kernel_m_read_readvariableop.savev2_adam_softmax_bias_m_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop0savev2_adam_softmax_kernel_v_read_readvariableop.savev2_adam_softmax_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *!
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:У
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2Й
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*Ђ
_input_shapes
: :::	Ш

:
: : : : : : : :::	Ш

:
:::	Ш

:
: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1:
 : : : : : : : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 

љ
+__inference_sequential_layer_call_fn_152730
conv1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallconv1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2*'
_output_shapes
:џџџџџџџџџ
*-
_gradient_op_typePartitionedCall-152723*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_152722
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :+ '
%
_user_specified_nameConv1_input: 

к
A__inference_Conv1_layer_call_and_return_conditional_losses_152604

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЊ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџj
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0Ѕ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
њ
_
C__inference_flatten_layer_call_and_return_conditional_losses_152630

inputs
identity^
Reshape/shapeConst*
valueB"џџџџH  *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ
Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ
"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs

г
F__inference_sequential_layer_call_and_return_conditional_losses_152722

inputs(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2*
&softmax_statefulpartitionedcall_args_1*
&softmax_statefulpartitionedcall_args_2
identityЂConv1/StatefulPartitionedCallЂSoftmax/StatefulPartitionedCall
Conv1/StatefulPartitionedCallStatefulPartitionedCallinputs$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџ*-
_gradient_op_typePartitionedCall-152610*J
fERC
A__inference_Conv1_layer_call_and_return_conditional_losses_152604Т
flatten/PartitionedCallPartitionedCall&Conv1/StatefulPartitionedCall:output:0*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_152630*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџШ
*-
_gradient_op_typePartitionedCall-152636
Softmax/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&softmax_statefulpartitionedcall_args_1&softmax_statefulpartitionedcall_args_2*'
_output_shapes
:џџџџџџџџџ
*
Tin
2*-
_gradient_op_typePartitionedCall-152660*L
fGRE
C__inference_Softmax_layer_call_and_return_conditional_losses_152654*
Tout
2**
config_proto

GPU 

CPU2J 8В
IdentityIdentity(Softmax/StatefulPartitionedCall:output:0^Conv1/StatefulPartitionedCall ^Softmax/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::2B
Softmax/StatefulPartitionedCallSoftmax/StatefulPartitionedCall2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
ж	
м
C__inference_Softmax_layer_call_and_return_conditional_losses_152654

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Ш

i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџШ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ќ

F__inference_sequential_layer_call_and_return_conditional_losses_152787

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource*
&softmax_matmul_readvariableop_resource+
'softmax_biasadd_readvariableop_resource
identityЂConv1/BiasAdd/ReadVariableOpЂConv1/Conv2D/ReadVariableOpЂSoftmax/BiasAdd/ReadVariableOpЂSoftmax/MatMul/ReadVariableOpЖ
Conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:І
Conv1/Conv2DConv2Dinputs#Conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*/
_output_shapes
:џџџџџџџџџ*
T0Ќ
Conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
Conv1/BiasAddBiasAddConv1/Conv2D:output:0$Conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџd

Conv1/ReluReluConv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџf
flatten/Reshape/shapeConst*
valueB"џџџџH  *
dtype0*
_output_shapes
:
flatten/ReshapeReshapeConv1/Relu:activations:0flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ
Г
Softmax/MatMul/ReadVariableOpReadVariableOp&softmax_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Ш


Softmax/MatMulMatMulflatten/Reshape:output:0%Softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
А
Softmax/BiasAdd/ReadVariableOpReadVariableOp'softmax_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:

Softmax/BiasAddBiasAddSoftmax/MatMul:product:0&Softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
f
Softmax/SoftmaxSoftmaxSoftmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
п
IdentityIdentitySoftmax/Softmax:softmax:0^Conv1/BiasAdd/ReadVariableOp^Conv1/Conv2D/ReadVariableOp^Softmax/BiasAdd/ReadVariableOp^Softmax/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::2>
Softmax/MatMul/ReadVariableOpSoftmax/MatMul/ReadVariableOp2<
Conv1/BiasAdd/ReadVariableOpConv1/BiasAdd/ReadVariableOp2:
Conv1/Conv2D/ReadVariableOpConv1/Conv2D/ReadVariableOp2@
Softmax/BiasAdd/ReadVariableOpSoftmax/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
ќ

F__inference_sequential_layer_call_and_return_conditional_losses_152767

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource*
&softmax_matmul_readvariableop_resource+
'softmax_biasadd_readvariableop_resource
identityЂConv1/BiasAdd/ReadVariableOpЂConv1/Conv2D/ReadVariableOpЂSoftmax/BiasAdd/ReadVariableOpЂSoftmax/MatMul/ReadVariableOpЖ
Conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:І
Conv1/Conv2DConv2Dinputs#Conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:џџџџџџџџџЌ
Conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
Conv1/BiasAddBiasAddConv1/Conv2D:output:0$Conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџd

Conv1/ReluReluConv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџf
flatten/Reshape/shapeConst*
valueB"џџџџH  *
dtype0*
_output_shapes
:
flatten/ReshapeReshapeConv1/Relu:activations:0flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ
Г
Softmax/MatMul/ReadVariableOpReadVariableOp&softmax_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Ш


Softmax/MatMulMatMulflatten/Reshape:output:0%Softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
А
Softmax/BiasAdd/ReadVariableOpReadVariableOp'softmax_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:

Softmax/BiasAddBiasAddSoftmax/MatMul:product:0&Softmax/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ
*
T0f
Softmax/SoftmaxSoftmaxSoftmax/BiasAdd:output:0*'
_output_shapes
:џџџџџџџџџ
*
T0п
IdentityIdentitySoftmax/Softmax:softmax:0^Conv1/BiasAdd/ReadVariableOp^Conv1/Conv2D/ReadVariableOp^Softmax/BiasAdd/ReadVariableOp^Softmax/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::2>
Softmax/MatMul/ReadVariableOpSoftmax/MatMul/ReadVariableOp2<
Conv1/BiasAdd/ReadVariableOpConv1/BiasAdd/ReadVariableOp2:
Conv1/Conv2D/ReadVariableOpConv1/Conv2D/ReadVariableOp2@
Softmax/BiasAdd/ReadVariableOpSoftmax/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
 
Ї
&__inference_Conv1_layer_call_fn_152615

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-152610*J
fERC
A__inference_Conv1_layer_call_and_return_conditional_losses_152604*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 

љ
+__inference_sequential_layer_call_fn_152707
conv1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallconv1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-152700*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_152699*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ
*
Tin	
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall: :+ '
%
_user_specified_nameConv1_input: : : 
ж	
м
C__inference_Softmax_layer_call_and_return_conditional_losses_152827

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Ш

i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ
*
T0"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџШ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
L
ш	
"__inference__traced_restore_152986
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias%
!assignvariableop_2_softmax_kernel#
assignvariableop_3_softmax_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count+
'assignvariableop_11_adam_conv1_kernel_m)
%assignvariableop_12_adam_conv1_bias_m-
)assignvariableop_13_adam_softmax_kernel_m+
'assignvariableop_14_adam_softmax_bias_m+
'assignvariableop_15_adam_conv1_kernel_v)
%assignvariableop_16_adam_conv1_bias_v-
)assignvariableop_17_adam_softmax_kernel_v+
'assignvariableop_18_adam_softmax_bias_v
identity_20ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1Д

RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*к	
valueа	BЭ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*9
value0B.B B B B B B B B B B B B B B B B B B B §
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*!
dtypes
2	*`
_output_shapesN
L:::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:y
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:}
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_softmax_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0
AssignVariableOp_3AssignVariableOpassignvariableop_3_softmax_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:|
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0*
dtype0	*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:~
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:~
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0}
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:x
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:{
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp'assignvariableop_11_adam_conv1_kernel_mIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0
AssignVariableOp_12AssignVariableOp%assignvariableop_12_adam_conv1_bias_mIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_softmax_kernel_mIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_softmax_bias_mIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_conv1_kernel_vIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_conv1_bias_vIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_softmax_kernel_vIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_softmax_bias_vIdentity_18:output:0*
dtype0*
_output_shapes
 
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:Е
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ё
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0ў
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2: : : : : :	 :
 : : : : : : : : : :+ '
%
_user_specified_namefile_prefix: : : 
і
Щ
!__inference__wrapped_model_152590
conv1_input3
/sequential_conv1_conv2d_readvariableop_resource4
0sequential_conv1_biasadd_readvariableop_resource5
1sequential_softmax_matmul_readvariableop_resource6
2sequential_softmax_biasadd_readvariableop_resource
identityЂ'sequential/Conv1/BiasAdd/ReadVariableOpЂ&sequential/Conv1/Conv2D/ReadVariableOpЂ)sequential/Softmax/BiasAdd/ReadVariableOpЂ(sequential/Softmax/MatMul/ReadVariableOpЬ
&sequential/Conv1/Conv2D/ReadVariableOpReadVariableOp/sequential_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:С
sequential/Conv1/Conv2DConv2Dconv1_input.sequential/Conv1/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingVALIDТ
'sequential/Conv1/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:А
sequential/Conv1/BiasAddBiasAdd sequential/Conv1/Conv2D:output:0/sequential/Conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџz
sequential/Conv1/ReluRelu!sequential/Conv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџq
 sequential/flatten/Reshape/shapeConst*
valueB"џџџџH  *
dtype0*
_output_shapes
:Ј
sequential/flatten/ReshapeReshape#sequential/Conv1/Relu:activations:0)sequential/flatten/Reshape/shape:output:0*(
_output_shapes
:џџџџџџџџџШ
*
T0Щ
(sequential/Softmax/MatMul/ReadVariableOpReadVariableOp1sequential_softmax_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Ш

Ќ
sequential/Softmax/MatMulMatMul#sequential/flatten/Reshape:output:00sequential/Softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
)sequential/Softmax/BiasAdd/ReadVariableOpReadVariableOp2sequential_softmax_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
Џ
sequential/Softmax/BiasAddBiasAdd#sequential/Softmax/MatMul:product:01sequential/Softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
|
sequential/Softmax/SoftmaxSoftmax#sequential/Softmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

IdentityIdentity$sequential/Softmax/Softmax:softmax:0(^sequential/Conv1/BiasAdd/ReadVariableOp'^sequential/Conv1/Conv2D/ReadVariableOp*^sequential/Softmax/BiasAdd/ReadVariableOp)^sequential/Softmax/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::2P
&sequential/Conv1/Conv2D/ReadVariableOp&sequential/Conv1/Conv2D/ReadVariableOp2R
'sequential/Conv1/BiasAdd/ReadVariableOp'sequential/Conv1/BiasAdd/ReadVariableOp2V
)sequential/Softmax/BiasAdd/ReadVariableOp)sequential/Softmax/BiasAdd/ReadVariableOp2T
(sequential/Softmax/MatMul/ReadVariableOp(sequential/Softmax/MatMul/ReadVariableOp: :+ '
%
_user_specified_nameConv1_input: : : 
х
ђ
$__inference_signature_wrapper_152745
conv1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-152738**
f%R#
!__inference__wrapped_model_152590*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2*'
_output_shapes
:џџџџџџџџџ

IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :+ '
%
_user_specified_nameConv1_input: 

є
+__inference_sequential_layer_call_fn_152796

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-152700*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_152699*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2*'
_output_shapes
:џџџџџџџџџ

IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
Н
D
(__inference_flatten_layer_call_fn_152816

inputs
identity
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџШ
*-
_gradient_op_typePartitionedCall-152636*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_152630a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ
"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
з
Љ
(__inference_Softmax_layer_call_fn_152834

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:џџџџџџџџџ
*-
_gradient_op_typePartitionedCall-152660*L
fGRE
C__inference_Softmax_layer_call_and_return_conditional_losses_152654*
Tout
2**
config_proto

GPU 

CPU2J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџШ
::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 

є
+__inference_sequential_layer_call_fn_152805

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2*'
_output_shapes
:џџџџџџџџџ
*-
_gradient_op_typePartitionedCall-152723*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_152722
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 

г
F__inference_sequential_layer_call_and_return_conditional_losses_152699

inputs(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2*
&softmax_statefulpartitionedcall_args_1*
&softmax_statefulpartitionedcall_args_2
identityЂConv1/StatefulPartitionedCallЂSoftmax/StatefulPartitionedCall
Conv1/StatefulPartitionedCallStatefulPartitionedCallinputs$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:џџџџџџџџџ*
Tin
2*-
_gradient_op_typePartitionedCall-152610*J
fERC
A__inference_Conv1_layer_call_and_return_conditional_losses_152604Т
flatten/PartitionedCallPartitionedCall&Conv1/StatefulPartitionedCall:output:0*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_152630*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџШ
*-
_gradient_op_typePartitionedCall-152636
Softmax/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0&softmax_statefulpartitionedcall_args_1&softmax_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-152660*L
fGRE
C__inference_Softmax_layer_call_and_return_conditional_losses_152654*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ
В
IdentityIdentity(Softmax/StatefulPartitionedCall:output:0^Conv1/StatefulPartitionedCall ^Softmax/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::2B
Softmax/StatefulPartitionedCallSoftmax/StatefulPartitionedCall2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*К
serving_defaultІ
K
Conv1_input<
serving_default_Conv1_input:0џџџџџџџџџ;
Softmax0
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:њ
Є
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
*L&call_and_return_all_conditional_losses
M_default_save_signature
N__call__"ё
_tf_keras_sequentialв{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "Conv1", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "Conv1", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Й
trainable_variables
	variables
regularization_losses
	keras_api
*O&call_and_return_all_conditional_losses
P__call__"Њ
_tf_keras_layer{"class_name": "InputLayer", "name": "Conv1_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28, 28, 1], "config": {"batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "sparse": false, "name": "Conv1_input"}}


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"ѕ
_tf_keras_layerл{"class_name": "Conv2D", "name": "Conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28, 28, 1], "config": {"name": "Conv1", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
Ќ
trainable_variables
	variables
regularization_losses
	keras_api
*S&call_and_return_all_conditional_losses
T__call__"
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
і

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*U&call_and_return_all_conditional_losses
V__call__"б
_tf_keras_layerЗ{"class_name": "Dense", "name": "Softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1352}}}}

iter

 beta_1

!beta_2
	"decay
#learning_ratemDmEmFmGvHvIvJvK"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
З

$layers
trainable_variables
	variables
%non_trainable_variables
&metrics
'layer_regularization_losses
regularization_losses
N__call__
M_default_save_signature
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
,
Wserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper


(layers
trainable_variables
	variables
)non_trainable_variables
*metrics
+layer_regularization_losses
regularization_losses
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
&:$2Conv1/kernel
:2
Conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper


,layers
trainable_variables
	variables
-non_trainable_variables
.metrics
/layer_regularization_losses
regularization_losses
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper


0layers
trainable_variables
	variables
1non_trainable_variables
2metrics
3layer_regularization_losses
regularization_losses
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
!:	Ш

2Softmax/kernel
:
2Softmax/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper


4layers
trainable_variables
	variables
5non_trainable_variables
6metrics
7layer_regularization_losses
regularization_losses
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

	9total
	:count
;
_fn_kwargs
<trainable_variables
=	variables
>regularization_losses
?	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"х
_tf_keras_layerЫ{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper


@layers
<trainable_variables
=	variables
Anon_trainable_variables
Bmetrics
Clayer_regularization_losses
>regularization_losses
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
+:)2Adam/Conv1/kernel/m
:2Adam/Conv1/bias/m
&:$	Ш

2Adam/Softmax/kernel/m
:
2Adam/Softmax/bias/m
+:)2Adam/Conv1/kernel/v
:2Adam/Conv1/bias/v
&:$	Ш

2Adam/Softmax/kernel/v
:
2Adam/Softmax/bias/v
ц2у
F__inference_sequential_layer_call_and_return_conditional_losses_152787
F__inference_sequential_layer_call_and_return_conditional_losses_152767
F__inference_sequential_layer_call_and_return_conditional_losses_152672
F__inference_sequential_layer_call_and_return_conditional_losses_152685Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ы2ш
!__inference__wrapped_model_152590Т
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *2Ђ/
-*
Conv1_inputџџџџџџџџџ
њ2ї
+__inference_sequential_layer_call_fn_152805
+__inference_sequential_layer_call_fn_152730
+__inference_sequential_layer_call_fn_152796
+__inference_sequential_layer_call_fn_152707Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
 2
A__inference_Conv1_layer_call_and_return_conditional_losses_152604з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_Conv1_layer_call_fn_152615з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
э2ъ
C__inference_flatten_layer_call_and_return_conditional_losses_152811Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_flatten_layer_call_fn_152816Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_Softmax_layer_call_and_return_conditional_losses_152827Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_Softmax_layer_call_fn_152834Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
7B5
$__inference_signature_wrapper_152745Conv1_input
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 ж
A__inference_Conv1_layer_call_and_return_conditional_losses_152604IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 И
F__inference_sequential_layer_call_and_return_conditional_losses_152767n?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 
+__inference_sequential_layer_call_fn_152805a?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
|
(__inference_Softmax_layer_call_fn_152834P0Ђ-
&Ђ#
!
inputsџџџџџџџџџШ

Њ "џџџџџџџџџ

!__inference__wrapped_model_152590w<Ђ9
2Ђ/
-*
Conv1_inputџџџџџџџџџ
Њ "1Њ.
,
Softmax!
Softmaxџџџџџџџџџ
Н
F__inference_sequential_layer_call_and_return_conditional_losses_152672sDЂA
:Ђ7
-*
Conv1_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 
+__inference_sequential_layer_call_fn_152707fDЂA
:Ђ7
-*
Conv1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

(__inference_flatten_layer_call_fn_152816T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџШ
И
F__inference_sequential_layer_call_and_return_conditional_losses_152787n?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Ў
&__inference_Conv1_layer_call_fn_152615IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЄ
C__inference_Softmax_layer_call_and_return_conditional_losses_152827]0Ђ-
&Ђ#
!
inputsџџџџџџџџџШ

Њ "%Ђ"

0џџџџџџџџџ

 Џ
$__inference_signature_wrapper_152745KЂH
Ђ 
AЊ>
<
Conv1_input-*
Conv1_inputџџџџџџџџџ"1Њ.
,
Softmax!
Softmaxџџџџџџџџџ
Н
F__inference_sequential_layer_call_and_return_conditional_losses_152685sDЂA
:Ђ7
-*
Conv1_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 
+__inference_sequential_layer_call_fn_152730fDЂA
:Ђ7
-*
Conv1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
Ј
C__inference_flatten_layer_call_and_return_conditional_losses_152811a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџШ

 
+__inference_sequential_layer_call_fn_152796a?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
