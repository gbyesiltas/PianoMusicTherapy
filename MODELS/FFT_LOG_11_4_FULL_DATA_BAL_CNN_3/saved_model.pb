??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
?
	MLCConv2D

input"T
filter"T

unique_key"T*num_args
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)
"
	transposebool( "
num_argsint(
?
	MLCMatMul
a"T
b"T

unique_key"T*num_args
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2"
num_argsint ("

input_rankint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
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
?
Select
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.4.0-rc02v1.12.1-44575-gc069d5bd9038??
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:#* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:#*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:
*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:
*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/m
?
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*'
shared_nameAdam/conv2d_5/kernel/m
?
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:#*
dtype0
?
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/m
?
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv2d_7/kernel/m
?
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d_7/bias/m
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/v
?
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*'
shared_nameAdam/conv2d_5/kernel/v
?
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:#*
dtype0
?
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/v
?
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv2d_7/kernel/v
?
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d_7/bias/v
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?G
value?GB?G B?G
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
 	variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
R
)trainable_variables
*	variables
+regularization_losses
,	keras_api
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
R
7trainable_variables
8	variables
9regularization_losses
:	keras_api
R
;trainable_variables
<	variables
=regularization_losses
>	keras_api
h

?kernel
@bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
R
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
R
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
R
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
h

Qkernel
Rbias
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
?
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratem?m?#m?$m?1m?2m??m?@m?Qm?Rm?v?v?#v?$v?1v?2v??v?@v?Qv?Rv?
F
0
1
#2
$3
14
25
?6
@7
Q8
R9
F
0
1
#2
$3
14
25
?6
@7
Q8
R9
 
?
\non_trainable_variables
]layer_metrics
trainable_variables
^layer_regularization_losses
_metrics

`layers
	variables
regularization_losses
 
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
anon_trainable_variables
blayer_metrics
trainable_variables
clayer_regularization_losses
dmetrics

elayers
	variables
regularization_losses
 
 
 
?
fnon_trainable_variables
glayer_metrics
trainable_variables
hlayer_regularization_losses
imetrics

jlayers
	variables
regularization_losses
 
 
 
?
knon_trainable_variables
llayer_metrics
trainable_variables
mlayer_regularization_losses
nmetrics

olayers
 	variables
!regularization_losses
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
pnon_trainable_variables
qlayer_metrics
%trainable_variables
rlayer_regularization_losses
smetrics

tlayers
&	variables
'regularization_losses
 
 
 
?
unon_trainable_variables
vlayer_metrics
)trainable_variables
wlayer_regularization_losses
xmetrics

ylayers
*	variables
+regularization_losses
 
 
 
?
znon_trainable_variables
{layer_metrics
-trainable_variables
|layer_regularization_losses
}metrics

~layers
.	variables
/regularization_losses
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
non_trainable_variables
?layer_metrics
3trainable_variables
 ?layer_regularization_losses
?metrics
?layers
4	variables
5regularization_losses
 
 
 
?
?non_trainable_variables
?layer_metrics
7trainable_variables
 ?layer_regularization_losses
?metrics
?layers
8	variables
9regularization_losses
 
 
 
?
?non_trainable_variables
?layer_metrics
;trainable_variables
 ?layer_regularization_losses
?metrics
?layers
<	variables
=regularization_losses
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
?
?non_trainable_variables
?layer_metrics
Atrainable_variables
 ?layer_regularization_losses
?metrics
?layers
B	variables
Cregularization_losses
 
 
 
?
?non_trainable_variables
?layer_metrics
Etrainable_variables
 ?layer_regularization_losses
?metrics
?layers
F	variables
Gregularization_losses
 
 
 
?
?non_trainable_variables
?layer_metrics
Itrainable_variables
 ?layer_regularization_losses
?metrics
?layers
J	variables
Kregularization_losses
 
 
 
?
?non_trainable_variables
?layer_metrics
Mtrainable_variables
 ?layer_regularization_losses
?metrics
?layers
N	variables
Oregularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1

Q0
R1
 
?
?non_trainable_variables
?layer_metrics
Strainable_variables
 ?layer_regularization_losses
?metrics
?layers
T	variables
Uregularization_losses
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
 
 
 
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
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
 
 
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_4_inputPlaceholder*/
_output_shapes
:?????????T*
dtype0*$
shape:?????????T
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_4_inputconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1677129
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_1677756
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_1677871??	
?

?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1676740

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_1677587

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?
~
)__inference_dense_1_layer_call_fn_1677628

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_16769202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

l
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_1676850

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????.
*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????.
2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????.
2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????.
2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_1677562

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1676901

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?
N
2__inference_gaussian_noise_4_layer_call_fn_1677362

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_16766052
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?5
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1677267

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource-
)dense_1_mlcmatmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/MLCMatMul/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2D	MLCConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R*
num_args *
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	R2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2D	MLCConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	0*
num_args *
paddingVALID*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	02
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	02
conv2d_5/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2D	MLCConv2Dconv2d_5/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
num_args *
paddingVALID*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
conv2d_6/Relu?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2D	MLCConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
*
num_args *
paddingVALID*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????.
2
conv2d_7/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshapeconv2d_7/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
 dense_1/MLCMatMul/ReadVariableOpReadVariableOp)dense_1_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_1/MLCMatMul/ReadVariableOp?
dense_1/MLCMatMul	MLCMatMulflatten_1/Reshape:output:0(dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MLCMatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MLCMatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
IdentityIdentitydense_1/Sigmoid:y:0 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/MLCMatMul/ReadVariableOp dense_1/MLCMatMul/ReadVariableOp:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?

*__inference_conv2d_7_layer_call_fn_1677547

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_16768232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
m
4__inference_gaussian_dropout_4_layer_call_fn_1677382

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_16766292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?

?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1676574

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????	R2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?C
?
"__inference__wrapped_model_1676559
conv2d_4_input8
4sequential_1_conv2d_4_conv2d_readvariableop_resource9
5sequential_1_conv2d_4_biasadd_readvariableop_resource8
4sequential_1_conv2d_5_conv2d_readvariableop_resource9
5sequential_1_conv2d_5_biasadd_readvariableop_resource8
4sequential_1_conv2d_6_conv2d_readvariableop_resource9
5sequential_1_conv2d_6_biasadd_readvariableop_resource8
4sequential_1_conv2d_7_conv2d_readvariableop_resource9
5sequential_1_conv2d_7_biasadd_readvariableop_resource:
6sequential_1_dense_1_mlcmatmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity??,sequential_1/conv2d_4/BiasAdd/ReadVariableOp?+sequential_1/conv2d_4/Conv2D/ReadVariableOp?,sequential_1/conv2d_5/BiasAdd/ReadVariableOp?+sequential_1/conv2d_5/Conv2D/ReadVariableOp?,sequential_1/conv2d_6/BiasAdd/ReadVariableOp?+sequential_1/conv2d_6/Conv2D/ReadVariableOp?,sequential_1/conv2d_7/BiasAdd/ReadVariableOp?+sequential_1/conv2d_7/Conv2D/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?-sequential_1/dense_1/MLCMatMul/ReadVariableOp?
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/conv2d_4/Conv2D/ReadVariableOp?
sequential_1/conv2d_4/Conv2D	MLCConv2Dconv2d_4_input3sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R*
num_args *
paddingVALID*
strides
2
sequential_1/conv2d_4/Conv2D?
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp?
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R2
sequential_1/conv2d_4/BiasAdd?
sequential_1/conv2d_4/ReluRelu&sequential_1/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	R2
sequential_1/conv2d_4/Relu?
+sequential_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02-
+sequential_1/conv2d_5/Conv2D/ReadVariableOp?
sequential_1/conv2d_5/Conv2D	MLCConv2D(sequential_1/conv2d_4/Relu:activations:03sequential_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	0*
num_args *
paddingVALID*
strides
2
sequential_1/conv2d_5/Conv2D?
,sequential_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_5/BiasAdd/ReadVariableOp?
sequential_1/conv2d_5/BiasAddBiasAdd%sequential_1/conv2d_5/Conv2D:output:04sequential_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	02
sequential_1/conv2d_5/BiasAdd?
sequential_1/conv2d_5/ReluRelu&sequential_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	02
sequential_1/conv2d_5/Relu?
+sequential_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/conv2d_6/Conv2D/ReadVariableOp?
sequential_1/conv2d_6/Conv2D	MLCConv2D(sequential_1/conv2d_5/Relu:activations:03sequential_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
num_args *
paddingVALID*
strides
2
sequential_1/conv2d_6/Conv2D?
,sequential_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_6/BiasAdd/ReadVariableOp?
sequential_1/conv2d_6/BiasAddBiasAdd%sequential_1/conv2d_6/Conv2D:output:04sequential_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
sequential_1/conv2d_6/BiasAdd?
sequential_1/conv2d_6/ReluRelu&sequential_1/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
sequential_1/conv2d_6/Relu?
+sequential_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02-
+sequential_1/conv2d_7/Conv2D/ReadVariableOp?
sequential_1/conv2d_7/Conv2D	MLCConv2D(sequential_1/conv2d_6/Relu:activations:03sequential_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
*
num_args *
paddingVALID*
strides
2
sequential_1/conv2d_7/Conv2D?
,sequential_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_1/conv2d_7/BiasAdd/ReadVariableOp?
sequential_1/conv2d_7/BiasAddBiasAdd%sequential_1/conv2d_7/Conv2D:output:04sequential_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
2
sequential_1/conv2d_7/BiasAdd?
sequential_1/conv2d_7/ReluRelu&sequential_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????.
2
sequential_1/conv2d_7/Relu?
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential_1/flatten_1/Const?
sequential_1/flatten_1/ReshapeReshape(sequential_1/conv2d_7/Relu:activations:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_1/flatten_1/Reshape?
-sequential_1/dense_1/MLCMatMul/ReadVariableOpReadVariableOp6sequential_1_dense_1_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-sequential_1/dense_1/MLCMatMul/ReadVariableOp?
sequential_1/dense_1/MLCMatMul	MLCMatMul'sequential_1/flatten_1/Reshape:output:05sequential_1/dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_1/dense_1/MLCMatMul?
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOp?
sequential_1/dense_1/BiasAddBiasAdd(sequential_1/dense_1/MLCMatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/dense_1/BiasAdd?
sequential_1/dense_1/SigmoidSigmoid%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_1/dense_1/Sigmoid?
IdentityIdentity sequential_1/dense_1/Sigmoid:y:0-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp-^sequential_1/conv2d_5/BiasAdd/ReadVariableOp,^sequential_1/conv2d_5/Conv2D/ReadVariableOp-^sequential_1/conv2d_6/BiasAdd/ReadVariableOp,^sequential_1/conv2d_6/Conv2D/ReadVariableOp-^sequential_1/conv2d_7/BiasAdd/ReadVariableOp,^sequential_1/conv2d_7/Conv2D/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_5/BiasAdd/ReadVariableOp,sequential_1/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_5/Conv2D/ReadVariableOp+sequential_1/conv2d_5/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_6/BiasAdd/ReadVariableOp,sequential_1/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_6/Conv2D/ReadVariableOp+sequential_1/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_7/BiasAdd/ReadVariableOp,sequential_1/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_7/Conv2D/ReadVariableOp+sequential_1/conv2d_7/Conv2D/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/MLCMatMul/ReadVariableOp-sequential_1/dense_1/MLCMatMul/ReadVariableOp:_ [
/
_output_shapes
:?????????T
(
_user_specified_nameconv2d_4_input
?

n
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_1677373

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????	R*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	R2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	R2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	R2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?N
?
 __inference__traced_save_1677756
file_prefix.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::#::::
:
:	?:: : : : : :::#::::
:
:	?::::#::::
:
:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:#: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:%	!

_output_shapes
:	?: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:#: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:%!

_output_shapes
:	?: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:#: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:
: !

_output_shapes
:
:%"!

_output_shapes
:	?: #

_output_shapes
::$

_output_shapes
: 
?

l
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_1676601

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????	R*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	R2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	R2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	R2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?H
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1676937
conv2d_4_input
conv2d_4_1676585
conv2d_4_1676587
conv2d_5_1676668
conv2d_5_1676670
conv2d_6_1676751
conv2d_6_1676753
conv2d_7_1676834
conv2d_7_1676836
dense_1_1676931
dense_1_1676933
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?*gaussian_dropout_4/StatefulPartitionedCall?*gaussian_dropout_5/StatefulPartitionedCall?*gaussian_dropout_6/StatefulPartitionedCall?*gaussian_dropout_7/StatefulPartitionedCall?(gaussian_noise_4/StatefulPartitionedCall?(gaussian_noise_5/StatefulPartitionedCall?(gaussian_noise_6/StatefulPartitionedCall?(gaussian_noise_7/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_1676585conv2d_4_1676587*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_16765742"
 conv2d_4/StatefulPartitionedCall?
(gaussian_noise_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_16766012*
(gaussian_noise_4/StatefulPartitionedCall?
*gaussian_dropout_4/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_4/StatefulPartitionedCall:output:0)^gaussian_noise_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_16766292,
*gaussian_dropout_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_4/StatefulPartitionedCall:output:0conv2d_5_1676668conv2d_5_1676670*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_16766572"
 conv2d_5/StatefulPartitionedCall?
(gaussian_noise_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0+^gaussian_dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_16766842*
(gaussian_noise_5/StatefulPartitionedCall?
*gaussian_dropout_5/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_5/StatefulPartitionedCall:output:0)^gaussian_noise_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_16767122,
*gaussian_dropout_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_5/StatefulPartitionedCall:output:0conv2d_6_1676751conv2d_6_1676753*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_16767402"
 conv2d_6/StatefulPartitionedCall?
(gaussian_noise_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0+^gaussian_dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_16767672*
(gaussian_noise_6/StatefulPartitionedCall?
*gaussian_dropout_6/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_6/StatefulPartitionedCall:output:0)^gaussian_noise_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_16767952,
*gaussian_dropout_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_6/StatefulPartitionedCall:output:0conv2d_7_1676834conv2d_7_1676836*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_16768232"
 conv2d_7/StatefulPartitionedCall?
(gaussian_noise_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0+^gaussian_dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_16768502*
(gaussian_noise_7/StatefulPartitionedCall?
*gaussian_dropout_7/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_7/StatefulPartitionedCall:output:0)^gaussian_noise_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_16768782,
*gaussian_dropout_7/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall3gaussian_dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_16769012
flatten_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1676931dense_1_1676933*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_16769202!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall+^gaussian_dropout_4/StatefulPartitionedCall+^gaussian_dropout_5/StatefulPartitionedCall+^gaussian_dropout_6/StatefulPartitionedCall+^gaussian_dropout_7/StatefulPartitionedCall)^gaussian_noise_4/StatefulPartitionedCall)^gaussian_noise_5/StatefulPartitionedCall)^gaussian_noise_6/StatefulPartitionedCall)^gaussian_noise_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*gaussian_dropout_4/StatefulPartitionedCall*gaussian_dropout_4/StatefulPartitionedCall2X
*gaussian_dropout_5/StatefulPartitionedCall*gaussian_dropout_5/StatefulPartitionedCall2X
*gaussian_dropout_6/StatefulPartitionedCall*gaussian_dropout_6/StatefulPartitionedCall2X
*gaussian_dropout_7/StatefulPartitionedCall*gaussian_dropout_7/StatefulPartitionedCall2T
(gaussian_noise_4/StatefulPartitionedCall(gaussian_noise_4/StatefulPartitionedCall2T
(gaussian_noise_5/StatefulPartitionedCall(gaussian_noise_5/StatefulPartitionedCall2T
(gaussian_noise_6/StatefulPartitionedCall(gaussian_noise_6/StatefulPartitionedCall2T
(gaussian_noise_7/StatefulPartitionedCall(gaussian_noise_7/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????T
(
_user_specified_nameconv2d_4_input
?
m
4__inference_gaussian_dropout_6_layer_call_fn_1677522

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_16767952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????022
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?

n
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_1676795

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????0*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????02
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????02
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????02
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_1676605

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?
P
4__inference_gaussian_dropout_4_layer_call_fn_1677387

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_16766332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?

n
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_1676712

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????	0*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	02
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	02
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	02
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	0:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?

l
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_1677348

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????	R*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	R2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	R2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	R2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?
N
2__inference_gaussian_noise_6_layer_call_fn_1677502

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_16767712
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?8
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1677079

inputs
conv2d_4_1677044
conv2d_4_1677046
conv2d_5_1677051
conv2d_5_1677053
conv2d_6_1677058
conv2d_6_1677060
conv2d_7_1677065
conv2d_7_1677067
dense_1_1677073
dense_1_1677075
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_1677044conv2d_4_1677046*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_16765742"
 conv2d_4/StatefulPartitionedCall?
 gaussian_noise_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_16766052"
 gaussian_noise_4/PartitionedCall?
"gaussian_dropout_4/PartitionedCallPartitionedCall)gaussian_noise_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_16766332$
"gaussian_dropout_4/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_4/PartitionedCall:output:0conv2d_5_1677051conv2d_5_1677053*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_16766572"
 conv2d_5/StatefulPartitionedCall?
 gaussian_noise_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_16766882"
 gaussian_noise_5/PartitionedCall?
"gaussian_dropout_5/PartitionedCallPartitionedCall)gaussian_noise_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_16767162$
"gaussian_dropout_5/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_5/PartitionedCall:output:0conv2d_6_1677058conv2d_6_1677060*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_16767402"
 conv2d_6/StatefulPartitionedCall?
 gaussian_noise_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_16767712"
 gaussian_noise_6/PartitionedCall?
"gaussian_dropout_6/PartitionedCallPartitionedCall)gaussian_noise_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_16767992$
"gaussian_dropout_6/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_6/PartitionedCall:output:0conv2d_7_1677065conv2d_7_1677067*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_16768232"
 conv2d_7/StatefulPartitionedCall?
 gaussian_noise_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_16768542"
 gaussian_noise_7/PartitionedCall?
"gaussian_dropout_7/PartitionedCallPartitionedCall)gaussian_noise_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_16768822$
"gaussian_dropout_7/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall+gaussian_dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_16769012
flatten_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1677073dense_1_1677075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_16769202!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?

*__inference_conv2d_4_layer_call_fn_1677337

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_16765742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????T::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?	
?
.__inference_sequential_1_layer_call_fn_1677102
conv2d_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_16770792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????T
(
_user_specified_nameconv2d_4_input
?

l
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_1677488

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????0*
dtype0*
seed???)*
seed2??]2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????02
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????02
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????02
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_1677447

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	0:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1677603

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_1676799

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
m
4__inference_gaussian_dropout_5_layer_call_fn_1677452

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_16767122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	022
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?

?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1676657

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	0*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????	02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	R::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_1676688

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	0:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?

n
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_1677513

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????0*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????02
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????02
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????02
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?

?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1677468

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?
N
2__inference_gaussian_noise_7_layer_call_fn_1677572

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_16768542
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_1677377

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?

l
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_1677418

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????	0*
dtype0*
seed???)*
seed2??N2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	02
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	02
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	02
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	0:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?

l
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_1677558

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????.
*
dtype0*
seed???)*
seed2?ǡ2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????.
2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????.
2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????.
2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?

n
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_1677583

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????.
*
dtype0*
seed???)*
seed2?ͷ2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????.
2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????.
2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????.
2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?

n
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_1676878

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????.
*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????.
2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????.
2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????.
2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?
P
4__inference_gaussian_dropout_6_layer_call_fn_1677527

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_16767992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
.__inference_sequential_1_layer_call_fn_1677292

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_16770162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_1676716

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	0:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?

?
D__inference_dense_1_layer_call_and_return_conditional_losses_1676920

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_1676771

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_1677352

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?
m
4__inference_gaussian_dropout_7_layer_call_fn_1677592

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_16768782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?	
?
.__inference_sequential_1_layer_call_fn_1677039
conv2d_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_16770162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????T
(
_user_specified_nameconv2d_4_input
?
k
2__inference_gaussian_noise_4_layer_call_fn_1677357

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_16766012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?
N
2__inference_gaussian_noise_5_layer_call_fn_1677432

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_16766882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	0:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_1677422

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	0:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?H
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1677016

inputs
conv2d_4_1676981
conv2d_4_1676983
conv2d_5_1676988
conv2d_5_1676990
conv2d_6_1676995
conv2d_6_1676997
conv2d_7_1677002
conv2d_7_1677004
dense_1_1677010
dense_1_1677012
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?*gaussian_dropout_4/StatefulPartitionedCall?*gaussian_dropout_5/StatefulPartitionedCall?*gaussian_dropout_6/StatefulPartitionedCall?*gaussian_dropout_7/StatefulPartitionedCall?(gaussian_noise_4/StatefulPartitionedCall?(gaussian_noise_5/StatefulPartitionedCall?(gaussian_noise_6/StatefulPartitionedCall?(gaussian_noise_7/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_1676981conv2d_4_1676983*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_16765742"
 conv2d_4/StatefulPartitionedCall?
(gaussian_noise_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_16766012*
(gaussian_noise_4/StatefulPartitionedCall?
*gaussian_dropout_4/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_4/StatefulPartitionedCall:output:0)^gaussian_noise_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_16766292,
*gaussian_dropout_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_4/StatefulPartitionedCall:output:0conv2d_5_1676988conv2d_5_1676990*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_16766572"
 conv2d_5/StatefulPartitionedCall?
(gaussian_noise_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0+^gaussian_dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_16766842*
(gaussian_noise_5/StatefulPartitionedCall?
*gaussian_dropout_5/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_5/StatefulPartitionedCall:output:0)^gaussian_noise_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_16767122,
*gaussian_dropout_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_5/StatefulPartitionedCall:output:0conv2d_6_1676995conv2d_6_1676997*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_16767402"
 conv2d_6/StatefulPartitionedCall?
(gaussian_noise_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0+^gaussian_dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_16767672*
(gaussian_noise_6/StatefulPartitionedCall?
*gaussian_dropout_6/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_6/StatefulPartitionedCall:output:0)^gaussian_noise_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_16767952,
*gaussian_dropout_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_6/StatefulPartitionedCall:output:0conv2d_7_1677002conv2d_7_1677004*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_16768232"
 conv2d_7/StatefulPartitionedCall?
(gaussian_noise_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0+^gaussian_dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_16768502*
(gaussian_noise_7/StatefulPartitionedCall?
*gaussian_dropout_7/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_7/StatefulPartitionedCall:output:0)^gaussian_noise_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_16768782,
*gaussian_dropout_7/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall3gaussian_dropout_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_16769012
flatten_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1677010dense_1_1677012*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_16769202!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall+^gaussian_dropout_4/StatefulPartitionedCall+^gaussian_dropout_5/StatefulPartitionedCall+^gaussian_dropout_6/StatefulPartitionedCall+^gaussian_dropout_7/StatefulPartitionedCall)^gaussian_noise_4/StatefulPartitionedCall)^gaussian_noise_5/StatefulPartitionedCall)^gaussian_noise_6/StatefulPartitionedCall)^gaussian_noise_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*gaussian_dropout_4/StatefulPartitionedCall*gaussian_dropout_4/StatefulPartitionedCall2X
*gaussian_dropout_5/StatefulPartitionedCall*gaussian_dropout_5/StatefulPartitionedCall2X
*gaussian_dropout_6/StatefulPartitionedCall*gaussian_dropout_6/StatefulPartitionedCall2X
*gaussian_dropout_7/StatefulPartitionedCall*gaussian_dropout_7/StatefulPartitionedCall2T
(gaussian_noise_4/StatefulPartitionedCall(gaussian_noise_4/StatefulPartitionedCall2T
(gaussian_noise_5/StatefulPartitionedCall(gaussian_noise_5/StatefulPartitionedCall2T
(gaussian_noise_6/StatefulPartitionedCall(gaussian_noise_6/StatefulPartitionedCall2T
(gaussian_noise_7/StatefulPartitionedCall(gaussian_noise_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_1676633

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_1676854

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?

n
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_1676629

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????	R*
dtype0*
seed???)*
seed2՚Q2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	R2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	R2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	R2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	R:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?
?
.__inference_sequential_1_layer_call_fn_1677317

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_16770792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1677129
conv2d_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_16765592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????T
(
_user_specified_nameconv2d_4_input
?

*__inference_conv2d_6_layer_call_fn_1677477

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_16767402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_1677517

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
k
2__inference_gaussian_noise_6_layer_call_fn_1677497

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_16767672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????022
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_1677492

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
P
4__inference_gaussian_dropout_7_layer_call_fn_1677597

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_16768822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?
G
+__inference_flatten_1_layer_call_fn_1677608

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_16769012
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?
P
4__inference_gaussian_dropout_5_layer_call_fn_1677457

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_16767162
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	0:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_1677871
file_prefix$
 assignvariableop_conv2d_4_kernel$
 assignvariableop_1_conv2d_4_bias&
"assignvariableop_2_conv2d_5_kernel$
 assignvariableop_3_conv2d_5_bias&
"assignvariableop_4_conv2d_6_kernel$
 assignvariableop_5_conv2d_6_bias&
"assignvariableop_6_conv2d_7_kernel$
 assignvariableop_7_conv2d_7_bias%
!assignvariableop_8_dense_1_kernel#
assignvariableop_9_dense_1_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate.
*assignvariableop_15_adam_conv2d_4_kernel_m,
(assignvariableop_16_adam_conv2d_4_bias_m.
*assignvariableop_17_adam_conv2d_5_kernel_m,
(assignvariableop_18_adam_conv2d_5_bias_m.
*assignvariableop_19_adam_conv2d_6_kernel_m,
(assignvariableop_20_adam_conv2d_6_bias_m.
*assignvariableop_21_adam_conv2d_7_kernel_m,
(assignvariableop_22_adam_conv2d_7_bias_m-
)assignvariableop_23_adam_dense_1_kernel_m+
'assignvariableop_24_adam_dense_1_bias_m.
*assignvariableop_25_adam_conv2d_4_kernel_v,
(assignvariableop_26_adam_conv2d_4_bias_v.
*assignvariableop_27_adam_conv2d_5_kernel_v,
(assignvariableop_28_adam_conv2d_5_bias_v.
*assignvariableop_29_adam_conv2d_6_kernel_v,
(assignvariableop_30_adam_conv2d_6_bias_v.
*assignvariableop_31_adam_conv2d_7_kernel_v,
(assignvariableop_32_adam_conv2d_7_bias_v-
)assignvariableop_33_adam_dense_1_kernel_v+
'assignvariableop_34_adam_dense_1_bias_v
identity_36??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_conv2d_4_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_conv2d_4_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv2d_5_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv2d_5_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_6_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_6_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_7_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_7_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_4_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_4_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_5_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_5_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_6_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_6_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_7_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_7_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35?
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ۉ
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1677226

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource-
)dense_1_mlcmatmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/MLCMatMul/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2D	MLCConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R*
num_args *
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	R2
conv2d_4/Relu{
gaussian_noise_4/ShapeShapeconv2d_4/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_4/Shape?
#gaussian_noise_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gaussian_noise_4/random_normal/mean?
%gaussian_noise_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gaussian_noise_4/random_normal/stddev?
3gaussian_noise_4/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_4/Shape:output:0*
T0*/
_output_shapes
:?????????	R*
dtype0*
seed???)*
seed2???25
3gaussian_noise_4/random_normal/RandomStandardNormal?
"gaussian_noise_4/random_normal/mulMul<gaussian_noise_4/random_normal/RandomStandardNormal:output:0.gaussian_noise_4/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	R2$
"gaussian_noise_4/random_normal/mul?
gaussian_noise_4/random_normalAdd&gaussian_noise_4/random_normal/mul:z:0,gaussian_noise_4/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	R2 
gaussian_noise_4/random_normal?
gaussian_noise_4/addAddV2conv2d_4/Relu:activations:0"gaussian_noise_4/random_normal:z:0*
T0*/
_output_shapes
:?????????	R2
gaussian_noise_4/add|
gaussian_dropout_4/ShapeShapegaussian_noise_4/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_4/Shape?
%gaussian_dropout_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gaussian_dropout_4/random_normal/mean?
'gaussian_dropout_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2)
'gaussian_dropout_4/random_normal/stddev?
5gaussian_dropout_4/random_normal/RandomStandardNormalRandomStandardNormal!gaussian_dropout_4/Shape:output:0*
T0*/
_output_shapes
:?????????	R*
dtype0*
seed???)*
seed2?ؐ27
5gaussian_dropout_4/random_normal/RandomStandardNormal?
$gaussian_dropout_4/random_normal/mulMul>gaussian_dropout_4/random_normal/RandomStandardNormal:output:00gaussian_dropout_4/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	R2&
$gaussian_dropout_4/random_normal/mul?
 gaussian_dropout_4/random_normalAdd(gaussian_dropout_4/random_normal/mul:z:0.gaussian_dropout_4/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	R2"
 gaussian_dropout_4/random_normal?
gaussian_dropout_4/mulMulgaussian_noise_4/add:z:0$gaussian_dropout_4/random_normal:z:0*
T0*/
_output_shapes
:?????????	R2
gaussian_dropout_4/mul?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2D	MLCConv2Dgaussian_dropout_4/mul:z:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	0*
num_args *
paddingVALID*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	02
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	02
conv2d_5/Relu{
gaussian_noise_5/ShapeShapeconv2d_5/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_5/Shape?
#gaussian_noise_5/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gaussian_noise_5/random_normal/mean?
%gaussian_noise_5/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gaussian_noise_5/random_normal/stddev?
3gaussian_noise_5/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_5/Shape:output:0*
T0*/
_output_shapes
:?????????	0*
dtype0*
seed???)*
seed2???25
3gaussian_noise_5/random_normal/RandomStandardNormal?
"gaussian_noise_5/random_normal/mulMul<gaussian_noise_5/random_normal/RandomStandardNormal:output:0.gaussian_noise_5/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	02$
"gaussian_noise_5/random_normal/mul?
gaussian_noise_5/random_normalAdd&gaussian_noise_5/random_normal/mul:z:0,gaussian_noise_5/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	02 
gaussian_noise_5/random_normal?
gaussian_noise_5/addAddV2conv2d_5/Relu:activations:0"gaussian_noise_5/random_normal:z:0*
T0*/
_output_shapes
:?????????	02
gaussian_noise_5/add|
gaussian_dropout_5/ShapeShapegaussian_noise_5/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_5/Shape?
%gaussian_dropout_5/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gaussian_dropout_5/random_normal/mean?
'gaussian_dropout_5/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2)
'gaussian_dropout_5/random_normal/stddev?
5gaussian_dropout_5/random_normal/RandomStandardNormalRandomStandardNormal!gaussian_dropout_5/Shape:output:0*
T0*/
_output_shapes
:?????????	0*
dtype0*
seed???)*
seed2?׻27
5gaussian_dropout_5/random_normal/RandomStandardNormal?
$gaussian_dropout_5/random_normal/mulMul>gaussian_dropout_5/random_normal/RandomStandardNormal:output:00gaussian_dropout_5/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	02&
$gaussian_dropout_5/random_normal/mul?
 gaussian_dropout_5/random_normalAdd(gaussian_dropout_5/random_normal/mul:z:0.gaussian_dropout_5/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	02"
 gaussian_dropout_5/random_normal?
gaussian_dropout_5/mulMulgaussian_noise_5/add:z:0$gaussian_dropout_5/random_normal:z:0*
T0*/
_output_shapes
:?????????	02
gaussian_dropout_5/mul?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2D	MLCConv2Dgaussian_dropout_5/mul:z:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
num_args *
paddingVALID*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
conv2d_6/Relu{
gaussian_noise_6/ShapeShapeconv2d_6/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_6/Shape?
#gaussian_noise_6/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gaussian_noise_6/random_normal/mean?
%gaussian_noise_6/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gaussian_noise_6/random_normal/stddev?
3gaussian_noise_6/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_6/Shape:output:0*
T0*/
_output_shapes
:?????????0*
dtype0*
seed???)*
seed2???25
3gaussian_noise_6/random_normal/RandomStandardNormal?
"gaussian_noise_6/random_normal/mulMul<gaussian_noise_6/random_normal/RandomStandardNormal:output:0.gaussian_noise_6/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????02$
"gaussian_noise_6/random_normal/mul?
gaussian_noise_6/random_normalAdd&gaussian_noise_6/random_normal/mul:z:0,gaussian_noise_6/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????02 
gaussian_noise_6/random_normal?
gaussian_noise_6/addAddV2conv2d_6/Relu:activations:0"gaussian_noise_6/random_normal:z:0*
T0*/
_output_shapes
:?????????02
gaussian_noise_6/add|
gaussian_dropout_6/ShapeShapegaussian_noise_6/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_6/Shape?
%gaussian_dropout_6/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gaussian_dropout_6/random_normal/mean?
'gaussian_dropout_6/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2)
'gaussian_dropout_6/random_normal/stddev?
5gaussian_dropout_6/random_normal/RandomStandardNormalRandomStandardNormal!gaussian_dropout_6/Shape:output:0*
T0*/
_output_shapes
:?????????0*
dtype0*
seed???)*
seed2Ȣ?27
5gaussian_dropout_6/random_normal/RandomStandardNormal?
$gaussian_dropout_6/random_normal/mulMul>gaussian_dropout_6/random_normal/RandomStandardNormal:output:00gaussian_dropout_6/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????02&
$gaussian_dropout_6/random_normal/mul?
 gaussian_dropout_6/random_normalAdd(gaussian_dropout_6/random_normal/mul:z:0.gaussian_dropout_6/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????02"
 gaussian_dropout_6/random_normal?
gaussian_dropout_6/mulMulgaussian_noise_6/add:z:0$gaussian_dropout_6/random_normal:z:0*
T0*/
_output_shapes
:?????????02
gaussian_dropout_6/mul?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2D	MLCConv2Dgaussian_dropout_6/mul:z:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
*
num_args *
paddingVALID*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????.
2
conv2d_7/Relu{
gaussian_noise_7/ShapeShapeconv2d_7/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_7/Shape?
#gaussian_noise_7/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gaussian_noise_7/random_normal/mean?
%gaussian_noise_7/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gaussian_noise_7/random_normal/stddev?
3gaussian_noise_7/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_7/Shape:output:0*
T0*/
_output_shapes
:?????????.
*
dtype0*
seed???)*
seed2??425
3gaussian_noise_7/random_normal/RandomStandardNormal?
"gaussian_noise_7/random_normal/mulMul<gaussian_noise_7/random_normal/RandomStandardNormal:output:0.gaussian_noise_7/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????.
2$
"gaussian_noise_7/random_normal/mul?
gaussian_noise_7/random_normalAdd&gaussian_noise_7/random_normal/mul:z:0,gaussian_noise_7/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????.
2 
gaussian_noise_7/random_normal?
gaussian_noise_7/addAddV2conv2d_7/Relu:activations:0"gaussian_noise_7/random_normal:z:0*
T0*/
_output_shapes
:?????????.
2
gaussian_noise_7/add|
gaussian_dropout_7/ShapeShapegaussian_noise_7/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_7/Shape?
%gaussian_dropout_7/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gaussian_dropout_7/random_normal/mean?
'gaussian_dropout_7/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'gaussian_dropout_7/random_normal/stddev?
5gaussian_dropout_7/random_normal/RandomStandardNormalRandomStandardNormal!gaussian_dropout_7/Shape:output:0*
T0*/
_output_shapes
:?????????.
*
dtype0*
seed???)*
seed2???27
5gaussian_dropout_7/random_normal/RandomStandardNormal?
$gaussian_dropout_7/random_normal/mulMul>gaussian_dropout_7/random_normal/RandomStandardNormal:output:00gaussian_dropout_7/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????.
2&
$gaussian_dropout_7/random_normal/mul?
 gaussian_dropout_7/random_normalAdd(gaussian_dropout_7/random_normal/mul:z:0.gaussian_dropout_7/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????.
2"
 gaussian_dropout_7/random_normal?
gaussian_dropout_7/mulMulgaussian_noise_7/add:z:0$gaussian_dropout_7/random_normal:z:0*
T0*/
_output_shapes
:?????????.
2
gaussian_dropout_7/muls
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshapegaussian_dropout_7/mul:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
 dense_1/MLCMatMul/ReadVariableOpReadVariableOp)dense_1_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_1/MLCMatMul/ReadVariableOp?
dense_1/MLCMatMul	MLCMatMulflatten_1/Reshape:output:0(dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MLCMatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MLCMatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
IdentityIdentitydense_1/Sigmoid:y:0 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/MLCMatMul/ReadVariableOp dense_1/MLCMatMul/ReadVariableOp:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?

l
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_1676684

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????	0*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	02
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	02
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	02
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	0:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_1676882

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?

?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1677328

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????	R2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????	R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?

n
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_1677443

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????	0*
dtype0*
seed???)*
seed2??g2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	02
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	02
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	02
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	0:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?

*__inference_conv2d_5_layer_call_fn_1677407

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_16766572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	R::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?
k
2__inference_gaussian_noise_7_layer_call_fn_1677567

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_16768502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????.
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????.

 
_user_specified_nameinputs
?

?
D__inference_dense_1_layer_call_and_return_conditional_losses_1677619

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1676823

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????.
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?

?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1677538

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????.
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????.
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?

?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1677398

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	0*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????	02
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	R::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????	R
 
_user_specified_nameinputs
?

l
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_1676767

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:?????????0*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????02
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????02
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????02
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
k
2__inference_gaussian_noise_5_layer_call_fn_1677427

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_16766842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	022
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	0
 
_user_specified_nameinputs
?8
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1676975
conv2d_4_input
conv2d_4_1676940
conv2d_4_1676942
conv2d_5_1676947
conv2d_5_1676949
conv2d_6_1676954
conv2d_6_1676956
conv2d_7_1676961
conv2d_7_1676963
dense_1_1676969
dense_1_1676971
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_1676940conv2d_4_1676942*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_16765742"
 conv2d_4/StatefulPartitionedCall?
 gaussian_noise_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_16766052"
 gaussian_noise_4/PartitionedCall?
"gaussian_dropout_4/PartitionedCallPartitionedCall)gaussian_noise_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_16766332$
"gaussian_dropout_4/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_4/PartitionedCall:output:0conv2d_5_1676947conv2d_5_1676949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_16766572"
 conv2d_5/StatefulPartitionedCall?
 gaussian_noise_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_16766882"
 gaussian_noise_5/PartitionedCall?
"gaussian_dropout_5/PartitionedCallPartitionedCall)gaussian_noise_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_16767162$
"gaussian_dropout_5/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_5/PartitionedCall:output:0conv2d_6_1676954conv2d_6_1676956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_16767402"
 conv2d_6/StatefulPartitionedCall?
 gaussian_noise_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_16767712"
 gaussian_noise_6/PartitionedCall?
"gaussian_dropout_6/PartitionedCallPartitionedCall)gaussian_noise_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_16767992$
"gaussian_dropout_6/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_6/PartitionedCall:output:0conv2d_7_1676961conv2d_7_1676963*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_16768232"
 conv2d_7/StatefulPartitionedCall?
 gaussian_noise_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_16768542"
 gaussian_noise_7/PartitionedCall?
"gaussian_dropout_7/PartitionedCallPartitionedCall)gaussian_noise_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????.
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_16768822$
"gaussian_dropout_7/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall+gaussian_dropout_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_16769012
flatten_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1676969dense_1_1676971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_16769202!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????T
(
_user_specified_nameconv2d_4_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
conv2d_4_input?
 serving_default_conv2d_4_input:0?????????T;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?T
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?P
_tf_keras_sequential?O{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_4", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [1, 35]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_5", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_6", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_6", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_7", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 84, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_4", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [1, 35]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_5", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_6", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_6", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_7", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 84, 1]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_4", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
trainable_variables
 	variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1}}
?


#kernel
$bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [1, 35]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 82, 30]}}
?
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_5", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1}}
?


1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 48, 30]}}
?
7trainable_variables
8	variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_6", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
;trainable_variables
<	variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_6", "trainable": true, "dtype": "float32", "rate": 0.1}}
?


?kernel
@bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 48, 30]}}
?
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_7", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5}}
?
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Qkernel
Rbias
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 460}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 460]}}
?
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratem?m?#m?$m?1m?2m??m?@m?Qm?Rm?v?v?#v?$v?1v?2v??v?@v?Qv?Rv?"
	optimizer
f
0
1
#2
$3
14
25
?6
@7
Q8
R9"
trackable_list_wrapper
f
0
1
#2
$3
14
25
?6
@7
Q8
R9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables
]layer_metrics
trainable_variables
^layer_regularization_losses
_metrics

`layers
	variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):'2conv2d_4/kernel
:2conv2d_4/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
anon_trainable_variables
blayer_metrics
trainable_variables
clayer_regularization_losses
dmetrics

elayers
	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables
glayer_metrics
trainable_variables
hlayer_regularization_losses
imetrics

jlayers
	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
knon_trainable_variables
llayer_metrics
trainable_variables
mlayer_regularization_losses
nmetrics

olayers
 	variables
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'#2conv2d_5/kernel
:2conv2d_5/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables
qlayer_metrics
%trainable_variables
rlayer_regularization_losses
smetrics

tlayers
&	variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables
vlayer_metrics
)trainable_variables
wlayer_regularization_losses
xmetrics

ylayers
*	variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
znon_trainable_variables
{layer_metrics
-trainable_variables
|layer_regularization_losses
}metrics

~layers
.	variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_6/kernel
:2conv2d_6/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
?layer_metrics
3trainable_variables
 ?layer_regularization_losses
?metrics
?layers
4	variables
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
7trainable_variables
 ?layer_regularization_losses
?metrics
?layers
8	variables
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
;trainable_variables
 ?layer_regularization_losses
?metrics
?layers
<	variables
=regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
2conv2d_7/kernel
:
2conv2d_7/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
Atrainable_variables
 ?layer_regularization_losses
?metrics
?layers
B	variables
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
Etrainable_variables
 ?layer_regularization_losses
?metrics
?layers
F	variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
Itrainable_variables
 ?layer_regularization_losses
?metrics
?layers
J	variables
Kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
Mtrainable_variables
 ?layer_regularization_losses
?metrics
?layers
N	variables
Oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_1/kernel
:2dense_1/bias
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
Strainable_variables
 ?layer_regularization_losses
?metrics
?layers
T	variables
Uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.:,2Adam/conv2d_4/kernel/m
 :2Adam/conv2d_4/bias/m
.:,#2Adam/conv2d_5/kernel/m
 :2Adam/conv2d_5/bias/m
.:,2Adam/conv2d_6/kernel/m
 :2Adam/conv2d_6/bias/m
.:,
2Adam/conv2d_7/kernel/m
 :
2Adam/conv2d_7/bias/m
&:$	?2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
.:,2Adam/conv2d_4/kernel/v
 :2Adam/conv2d_4/bias/v
.:,#2Adam/conv2d_5/kernel/v
 :2Adam/conv2d_5/bias/v
.:,2Adam/conv2d_6/kernel/v
 :2Adam/conv2d_6/bias/v
.:,
2Adam/conv2d_7/kernel/v
 :
2Adam/conv2d_7/bias/v
&:$	?2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
?2?
.__inference_sequential_1_layer_call_fn_1677102
.__inference_sequential_1_layer_call_fn_1677317
.__inference_sequential_1_layer_call_fn_1677292
.__inference_sequential_1_layer_call_fn_1677039?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_1676559?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *5?2
0?-
conv2d_4_input?????????T
?2?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1677226
I__inference_sequential_1_layer_call_and_return_conditional_losses_1676937
I__inference_sequential_1_layer_call_and_return_conditional_losses_1677267
I__inference_sequential_1_layer_call_and_return_conditional_losses_1676975?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_4_layer_call_fn_1677337?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1677328?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_gaussian_noise_4_layer_call_fn_1677357
2__inference_gaussian_noise_4_layer_call_fn_1677362?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_1677352
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_1677348?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_gaussian_dropout_4_layer_call_fn_1677382
4__inference_gaussian_dropout_4_layer_call_fn_1677387?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_1677373
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_1677377?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_5_layer_call_fn_1677407?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1677398?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_gaussian_noise_5_layer_call_fn_1677427
2__inference_gaussian_noise_5_layer_call_fn_1677432?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_1677422
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_1677418?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_gaussian_dropout_5_layer_call_fn_1677457
4__inference_gaussian_dropout_5_layer_call_fn_1677452?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_1677447
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_1677443?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_6_layer_call_fn_1677477?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1677468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_gaussian_noise_6_layer_call_fn_1677502
2__inference_gaussian_noise_6_layer_call_fn_1677497?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_1677488
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_1677492?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_gaussian_dropout_6_layer_call_fn_1677527
4__inference_gaussian_dropout_6_layer_call_fn_1677522?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_1677513
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_1677517?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_7_layer_call_fn_1677547?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1677538?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_gaussian_noise_7_layer_call_fn_1677572
2__inference_gaussian_noise_7_layer_call_fn_1677567?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_1677562
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_1677558?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_gaussian_dropout_7_layer_call_fn_1677592
4__inference_gaussian_dropout_7_layer_call_fn_1677597?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_1677583
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_1677587?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_flatten_1_layer_call_fn_1677608?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_1_layer_call_and_return_conditional_losses_1677603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_1_layer_call_fn_1677628?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_1677619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_1677129conv2d_4_input"?
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
 ?
"__inference__wrapped_model_1676559?
#$12?@QR??<
5?2
0?-
conv2d_4_input?????????T
? "1?.
,
dense_1!?
dense_1??????????
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1677328l7?4
-?*
(?%
inputs?????????T
? "-?*
#? 
0?????????	R
? ?
*__inference_conv2d_4_layer_call_fn_1677337_7?4
-?*
(?%
inputs?????????T
? " ??????????	R?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1677398l#$7?4
-?*
(?%
inputs?????????	R
? "-?*
#? 
0?????????	0
? ?
*__inference_conv2d_5_layer_call_fn_1677407_#$7?4
-?*
(?%
inputs?????????	R
? " ??????????	0?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1677468l127?4
-?*
(?%
inputs?????????	0
? "-?*
#? 
0?????????0
? ?
*__inference_conv2d_6_layer_call_fn_1677477_127?4
-?*
(?%
inputs?????????	0
? " ??????????0?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1677538l?@7?4
-?*
(?%
inputs?????????0
? "-?*
#? 
0?????????.

? ?
*__inference_conv2d_7_layer_call_fn_1677547_?@7?4
-?*
(?%
inputs?????????0
? " ??????????.
?
D__inference_dense_1_layer_call_and_return_conditional_losses_1677619]QR0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_1_layer_call_fn_1677628PQR0?-
&?#
!?
inputs??????????
? "???????????
F__inference_flatten_1_layer_call_and_return_conditional_losses_1677603a7?4
-?*
(?%
inputs?????????.

? "&?#
?
0??????????
? ?
+__inference_flatten_1_layer_call_fn_1677608T7?4
-?*
(?%
inputs?????????.

? "????????????
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_1677373l;?8
1?.
(?%
inputs?????????	R
p
? "-?*
#? 
0?????????	R
? ?
O__inference_gaussian_dropout_4_layer_call_and_return_conditional_losses_1677377l;?8
1?.
(?%
inputs?????????	R
p 
? "-?*
#? 
0?????????	R
? ?
4__inference_gaussian_dropout_4_layer_call_fn_1677382_;?8
1?.
(?%
inputs?????????	R
p
? " ??????????	R?
4__inference_gaussian_dropout_4_layer_call_fn_1677387_;?8
1?.
(?%
inputs?????????	R
p 
? " ??????????	R?
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_1677443l;?8
1?.
(?%
inputs?????????	0
p
? "-?*
#? 
0?????????	0
? ?
O__inference_gaussian_dropout_5_layer_call_and_return_conditional_losses_1677447l;?8
1?.
(?%
inputs?????????	0
p 
? "-?*
#? 
0?????????	0
? ?
4__inference_gaussian_dropout_5_layer_call_fn_1677452_;?8
1?.
(?%
inputs?????????	0
p
? " ??????????	0?
4__inference_gaussian_dropout_5_layer_call_fn_1677457_;?8
1?.
(?%
inputs?????????	0
p 
? " ??????????	0?
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_1677513l;?8
1?.
(?%
inputs?????????0
p
? "-?*
#? 
0?????????0
? ?
O__inference_gaussian_dropout_6_layer_call_and_return_conditional_losses_1677517l;?8
1?.
(?%
inputs?????????0
p 
? "-?*
#? 
0?????????0
? ?
4__inference_gaussian_dropout_6_layer_call_fn_1677522_;?8
1?.
(?%
inputs?????????0
p
? " ??????????0?
4__inference_gaussian_dropout_6_layer_call_fn_1677527_;?8
1?.
(?%
inputs?????????0
p 
? " ??????????0?
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_1677583l;?8
1?.
(?%
inputs?????????.

p
? "-?*
#? 
0?????????.

? ?
O__inference_gaussian_dropout_7_layer_call_and_return_conditional_losses_1677587l;?8
1?.
(?%
inputs?????????.

p 
? "-?*
#? 
0?????????.

? ?
4__inference_gaussian_dropout_7_layer_call_fn_1677592_;?8
1?.
(?%
inputs?????????.

p
? " ??????????.
?
4__inference_gaussian_dropout_7_layer_call_fn_1677597_;?8
1?.
(?%
inputs?????????.

p 
? " ??????????.
?
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_1677348l;?8
1?.
(?%
inputs?????????	R
p
? "-?*
#? 
0?????????	R
? ?
M__inference_gaussian_noise_4_layer_call_and_return_conditional_losses_1677352l;?8
1?.
(?%
inputs?????????	R
p 
? "-?*
#? 
0?????????	R
? ?
2__inference_gaussian_noise_4_layer_call_fn_1677357_;?8
1?.
(?%
inputs?????????	R
p
? " ??????????	R?
2__inference_gaussian_noise_4_layer_call_fn_1677362_;?8
1?.
(?%
inputs?????????	R
p 
? " ??????????	R?
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_1677418l;?8
1?.
(?%
inputs?????????	0
p
? "-?*
#? 
0?????????	0
? ?
M__inference_gaussian_noise_5_layer_call_and_return_conditional_losses_1677422l;?8
1?.
(?%
inputs?????????	0
p 
? "-?*
#? 
0?????????	0
? ?
2__inference_gaussian_noise_5_layer_call_fn_1677427_;?8
1?.
(?%
inputs?????????	0
p
? " ??????????	0?
2__inference_gaussian_noise_5_layer_call_fn_1677432_;?8
1?.
(?%
inputs?????????	0
p 
? " ??????????	0?
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_1677488l;?8
1?.
(?%
inputs?????????0
p
? "-?*
#? 
0?????????0
? ?
M__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_1677492l;?8
1?.
(?%
inputs?????????0
p 
? "-?*
#? 
0?????????0
? ?
2__inference_gaussian_noise_6_layer_call_fn_1677497_;?8
1?.
(?%
inputs?????????0
p
? " ??????????0?
2__inference_gaussian_noise_6_layer_call_fn_1677502_;?8
1?.
(?%
inputs?????????0
p 
? " ??????????0?
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_1677558l;?8
1?.
(?%
inputs?????????.

p
? "-?*
#? 
0?????????.

? ?
M__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_1677562l;?8
1?.
(?%
inputs?????????.

p 
? "-?*
#? 
0?????????.

? ?
2__inference_gaussian_noise_7_layer_call_fn_1677567_;?8
1?.
(?%
inputs?????????.

p
? " ??????????.
?
2__inference_gaussian_noise_7_layer_call_fn_1677572_;?8
1?.
(?%
inputs?????????.

p 
? " ??????????.
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1676937|
#$12?@QRG?D
=?:
0?-
conv2d_4_input?????????T
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1676975|
#$12?@QRG?D
=?:
0?-
conv2d_4_input?????????T
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1677226t
#$12?@QR??<
5?2
(?%
inputs?????????T
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1677267t
#$12?@QR??<
5?2
(?%
inputs?????????T
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_1_layer_call_fn_1677039o
#$12?@QRG?D
=?:
0?-
conv2d_4_input?????????T
p

 
? "???????????
.__inference_sequential_1_layer_call_fn_1677102o
#$12?@QRG?D
=?:
0?-
conv2d_4_input?????????T
p 

 
? "???????????
.__inference_sequential_1_layer_call_fn_1677292g
#$12?@QR??<
5?2
(?%
inputs?????????T
p

 
? "???????????
.__inference_sequential_1_layer_call_fn_1677317g
#$12?@QR??<
5?2
(?%
inputs?????????T
p 

 
? "???????????
%__inference_signature_wrapper_1677129?
#$12?@QRQ?N
? 
G?D
B
conv2d_4_input0?-
conv2d_4_input?????????T"1?.
,
dense_1!?
dense_1?????????