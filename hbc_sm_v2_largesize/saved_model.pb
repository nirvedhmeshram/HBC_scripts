??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
+
Ceil
x"T
y"T"
Ttype:
2
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
s
	ScatterNd
indices"Tindices
updates"T
shape"Tindices
output"T"	
Ttype"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28?c
Z
ConstConst*
_output_shapes

:؏*
dtype0*
valueB؏2        
\
Const_1Const*
_output_shapes

:؏*
dtype0*
valueB؏2        

NoOpNoOp
k
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*%
valueB B


signatures
 
h
serving_default_logitPlaceholder*
_output_shapes
:	?'*
dtype0*
shape:	?'
r
serving_default_segment_lengthsPlaceholder*
_output_shapes
:	?'*
dtype0	*
shape:	?'
h
serving_default_segment_valuePlaceholder*
_output_shapes	
:?'*
dtype0	*
shape:?'
?
PartitionedCallPartitionedCallserving_default_logitserving_default_segment_lengthsserving_default_segment_valueConstConst_1*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference_signature_wrapper_198
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference__traced_save_225
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_restore_235?V
?7
?
__inference___call___185
segment_value	
segment_lengths		
logit
gatherv2_params
gatherv2_1_params
identityJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *?j?K
subSublogitsub/y:output:0*
T0*
_output_shapes
:	?'E
SigmoidSigmoidsub:z:0*
T0*
_output_shapes
:	?'`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????]
ReshapeReshapeSigmoid:y:0Reshape/shape:output:0*
T0*
_output_shapes	
:?'G
SizeConst*
_output_shapes
: *
dtype0*
value
B :?'f
zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
zeros/ReshapeReshapeSize:output:0zeros/Reshape/shape:output:0*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R a
zerosFillzeros/Reshape:output:0zeros/Const:output:0*
T0	*
_output_shapes	
:?'I
Size_1Const*
_output_shapes
: *
dtype0*
value
B :?'M
range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R M
range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 RS

range/CastCastSize_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: s
rangeRangerange/start:output:0range/Cast:y:0range/delta:output:0*

Tidx0	*
_output_shapes	
:?'P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :k

ExpandDims
ExpandDimsrange:output:0ExpandDims/dim:output:0*
T0	*
_output_shapes
:	?'t
GatherNdGatherNdsegment_valueExpandDims:output:0*
Tindices0	*
Tparams0	*
_output_shapes	
:?'G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RU
addAddV2GatherNd:output:0add/y:output:0*
T0	*
_output_shapes	
:?'Z
ScatterNd/shapeConst*
_output_shapes
:*
dtype0	*
valueB	R?'?
	ScatterNd	ScatterNdExpandDims:output:0add:z:0ScatterNd/shape:output:0*
T0	*
Tindices0	*
_output_shapes	
:?'b
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????e
	Reshape_1Reshapesegment_lengthsReshape_1/shape:output:0*
T0	*
_output_shapes	
:?'b
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
	Reshape_2ReshapeScatterNd:output:0Reshape_2/shape:output:0*
T0	*
_output_shapes	
:?'b
zeros_1/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:?'O
zeros_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R o
zeros_1Fill zeros_1/shape_as_tensor:output:0zeros_1/Const:output:0*
T0	*
_output_shapes	
:?'_
ones/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:?'L

ones/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rf
onesFillones/shape_as_tensor:output:0ones/Const:output:0*
T0	*
_output_shapes	
:?'K
	Greater/yConst*
_output_shapes
: *
dtype0	*
value	B	 R*`
GreaterGreaterReshape_2:output:0Greater/y:output:0*
T0	*
_output_shapes	
:?'`
	Greater_1Greaterzeros_1:output:0Reshape_2:output:0*
T0	*
_output_shapes	
:?'w
EqualEqualReshape_1:output:0ones:output:0*
T0	*
_output_shapes	
:?'*
incompatible_shape_error( m
SelectV2SelectV2Greater:z:0zeros_1:output:0Reshape_2:output:0*
T0	*
_output_shapes	
:?'p

SelectV2_1SelectV2Greater_1:z:0zeros_1:output:0SelectV2:output:0*
T0	*
_output_shapes	
:?'n

SelectV2_2SelectV2	Equal:z:0SelectV2_1:output:0zeros_1:output:0*
T0	*
_output_shapes	
:?'N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Q9^
truedivRealDivReshape:output:0truediv/y:output:0*
T0*
_output_shapes	
:?'?
CeilCeiltruediv:z:0*
T0*
_output_shapes	
:?'L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??N
sub_1SubCeil:y:0sub_1/y:output:0*
T0*
_output_shapes	
:?'L
CastCast	sub_1:z:0*

DstT0	*

SrcT0*
_output_shapes	
:?'H
mul/yConst*
_output_shapes
: *
dtype0	*
value
B	 R?'U
mulMulSelectV2_2:output:0mul/y:output:0*
T0	*
_output_shapes	
:?'G
add_1AddV2Cast:y:0mul:z:0*
T0	*
_output_shapes	
:?'N
Cast_1Cast	add_1:z:0*

DstT0*

SrcT0	*
_output_shapes	
:?'b
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????`
	Reshape_3Reshape
Cast_1:y:0Reshape_3/shape:output:0*
T0*
_output_shapes	
:?'W
Cast_2CastReshape_3:output:0*

DstT0	*

SrcT0*
_output_shapes	
:?'O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2gatherv2_params
Cast_2:y:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes	
:?'W
Cast_3CastReshape_3:output:0*

DstT0	*

SrcT0*
_output_shapes	
:?'Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?

GatherV2_1GatherV2gatherv2_1_params
Cast_3:y:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes	
:?'b
	truediv_1RealDivGatherV2:output:0GatherV2_1:output:0*
T0*
_output_shapes	
:?'R
Cast_4Casttruediv_1:z:0*

DstT0*

SrcT0*
_output_shapes	
:?'L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *;??P
mul_1Mul
Cast_4:y:0mul_1/y:output:0*
T0*
_output_shapes	
:?'L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:V
mul_2MulReshape:output:0mul_2/y:output:0*
T0*
_output_shapes	
:?'J
add_2AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes	
:?'T
Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB 2     ??@e
	Greater_2GreaterGatherV2_1:output:0Greater_2/y:output:0*
T0*
_output_shapes	
:?'h

SelectV2_3SelectV2Greater_2:z:0	add_2:z:0Reshape:output:0*
T0*
_output_shapes	
:?'`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   m
	Reshape_4ReshapeSelectV2_3:output:0Reshape_4/shape:output:0*
T0*
_output_shapes
:	?'R
IdentityIdentityReshape_4:output:0*
T0*
_output_shapes
:	?'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?':	?':	?':؏:؏:J F

_output_shapes	
:?'
'
_user_specified_namesegment_value:PL

_output_shapes
:	?'
)
_user_specified_namesegment_lengths:FB

_output_shapes
:	?'

_user_specified_namelogit:"

_output_shapes

:؏:"

_output_shapes

:؏
?
E
__inference__traced_restore_235
file_prefix

identity_1??
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
k
__inference__traced_save_225
file_prefix
savev2_const_2

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const_2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
?
?
!__inference_signature_wrapper_198	
logit
segment_lengths	
segment_value	
unknown
	unknown_0
identity?
PartitionedCallPartitionedCallsegment_valuesegment_lengthslogitunknown	unknown_0*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *!
fR
__inference___call___185X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:	?':	?':?':؏:؏:F B

_output_shapes
:	?'

_user_specified_namelogit:PL

_output_shapes
:	?'
)
_user_specified_namesegment_lengths:JF

_output_shapes	
:?'
'
_user_specified_namesegment_value:"

_output_shapes

:؏:"

_output_shapes

:؏"?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
/
logit&
serving_default_logit:0	?'
C
segment_lengths0
!serving_default_segment_lengths:0		?'
;
segment_value*
serving_default_segment_value:0	?',
output_0 
PartitionedCall:0	?'tensorflow/serving/predict:?
<

signatures
__call__"
_generic_user_object
,
serving_default"
signature_map
?2?
__inference___call___185?
???
FullArgSpec@
args8?5
jself
jsegment_value
jsegment_lengths
jlogit
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *5?2
?	?'	
?	?'	
?	?'
?B?
!__inference_signature_wrapper_198logitsegment_lengthssegment_value"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
	J
Const
J	
Const_1?
__inference___call___185~f?c
\?Y
?
segment_value?'	
!?
segment_lengths	?'	
?
logit	?'
? "?	?'?
!__inference_signature_wrapper_198????
? 
???
 
logit?
logit	?'
4
segment_lengths!?
segment_lengths	?'	
,
segment_value?
segment_value?'	"+?(
&
output_0?
output_0	?'