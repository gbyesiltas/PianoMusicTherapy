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
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0
?
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:#* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:#*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:*
dtype0
?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:
*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:
*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	?*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
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
Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_8/kernel/m
?
*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_8/bias/m
y
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*'
shared_nameAdam/conv2d_9/kernel/m
?
*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*&
_output_shapes
:#*
dtype0
?
Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_9/bias/m
y
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_10/kernel/m
?
+Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_10/bias/m
{
)Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/conv2d_11/kernel/m
?
+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_11/bias/m
{
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_2/kernel/m
?
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_8/kernel/v
?
*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_8/bias/v
y
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*'
shared_nameAdam/conv2d_9/kernel/v
?
*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*&
_output_shapes
:#*
dtype0
?
Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_9/bias/v
y
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_10/kernel/v
?
+Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_10/bias/v
{
)Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/conv2d_11/kernel/v
?
+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_11/bias/v
{
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_2/kernel/v
?
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
R
)regularization_losses
*	variables
+trainable_variables
,	keras_api
R
-regularization_losses
.	variables
/trainable_variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
R
;regularization_losses
<	variables
=trainable_variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
R
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
R
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
R
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
h

Qkernel
Rbias
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
?
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratem?m?#m?$m?1m?2m??m?@m?Qm?Rm?v?v?#v?$v?1v?2v??v?@v?Qv?Rv?
 
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
?
\metrics
regularization_losses
	variables
]layer_regularization_losses
^layer_metrics
_non_trainable_variables

`layers
trainable_variables
 
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
ametrics
regularization_losses
	variables
blayer_regularization_losses
clayer_metrics
dnon_trainable_variables
trainable_variables

elayers
 
 
 
?
fmetrics
regularization_losses
	variables
glayer_regularization_losses
hlayer_metrics
inon_trainable_variables
trainable_variables

jlayers
 
 
 
?
kmetrics
regularization_losses
 	variables
llayer_regularization_losses
mlayer_metrics
nnon_trainable_variables
!trainable_variables

olayers
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
?
pmetrics
%regularization_losses
&	variables
qlayer_regularization_losses
rlayer_metrics
snon_trainable_variables
'trainable_variables

tlayers
 
 
 
?
umetrics
)regularization_losses
*	variables
vlayer_regularization_losses
wlayer_metrics
xnon_trainable_variables
+trainable_variables

ylayers
 
 
 
?
zmetrics
-regularization_losses
.	variables
{layer_regularization_losses
|layer_metrics
}non_trainable_variables
/trainable_variables

~layers
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
?
metrics
3regularization_losses
4	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
5trainable_variables
?layers
 
 
 
?
?metrics
7regularization_losses
8	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
9trainable_variables
?layers
 
 
 
?
?metrics
;regularization_losses
<	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
=trainable_variables
?layers
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
?
?metrics
Aregularization_losses
B	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Ctrainable_variables
?layers
 
 
 
?
?metrics
Eregularization_losses
F	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Gtrainable_variables
?layers
 
 
 
?
?metrics
Iregularization_losses
J	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Ktrainable_variables
?layers
 
 
 
?
?metrics
Mregularization_losses
N	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Otrainable_variables
?layers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
?
?metrics
Sregularization_losses
T	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Utrainable_variables
?layers
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
VARIABLE_VALUEAdam/conv2d_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_9/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_9/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_10/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_10/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_11/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_11/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_9/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_9/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_10/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_10/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_11/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_11/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_8_inputPlaceholder*/
_output_shapes
:?????????`*
dtype0*$
shape:?????????`
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_8_inputconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_2/kerneldense_2/bias*
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
%__inference_signature_wrapper_2350034
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp*Adam/conv2d_8/kernel/m/Read/ReadVariableOp(Adam/conv2d_8/bias/m/Read/ReadVariableOp*Adam/conv2d_9/kernel/m/Read/ReadVariableOp(Adam/conv2d_9/bias/m/Read/ReadVariableOp+Adam/conv2d_10/kernel/m/Read/ReadVariableOp)Adam/conv2d_10/bias/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp*Adam/conv2d_8/kernel/v/Read/ReadVariableOp(Adam/conv2d_8/bias/v/Read/ReadVariableOp*Adam/conv2d_9/kernel/v/Read/ReadVariableOp(Adam/conv2d_9/bias/v/Read/ReadVariableOp+Adam/conv2d_10/kernel/v/Read/ReadVariableOp)Adam/conv2d_10/bias/v/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*0
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
 __inference__traced_save_2350661
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/conv2d_10/kernel/mAdam/conv2d_10/bias/mAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/conv2d_10/kernel/vAdam/conv2d_10/bias/vAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*/
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
#__inference__traced_restore_2350776??	
?

n
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_2349617

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
:?????????	<*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	<2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	<2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	<2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?

o
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_2349783

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
:?????????:
*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????:
2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????:
2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????:
2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?
l
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_2350492

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?5
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2350172

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource-
)dense_2_mlcmatmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/MLCMatMul/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2D	MLCConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	^*
num_args *
paddingVALID*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	^2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	^2
conv2d_8/Relu?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2D	MLCConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	<*
num_args *
paddingVALID*
strides
2
conv2d_9/Conv2D?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	<2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	<2
conv2d_9/Relu?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2D	MLCConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
num_args *
paddingVALID*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
conv2d_10/Relu?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2D	MLCConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????:
*
num_args *
paddingVALID*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????:
2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????:
2
conv2d_11/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????D  2
flatten_2/Const?
flatten_2/ReshapeReshapeconv2d_11/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_2/Reshape?
 dense_2/MLCMatMul/ReadVariableOpReadVariableOp)dense_2_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_2/MLCMatMul/ReadVariableOp?
dense_2/MLCMatMul	MLCMatMulflatten_2/Reshape:output:0(dense_2/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MLCMatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MLCMatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Sigmoid?
IdentityIdentitydense_2/Sigmoid:y:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/MLCMatMul/ReadVariableOp dense_2/MLCMatMul/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_2349621

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?
n
5__inference_gaussian_dropout_11_layer_call_fn_2350497

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
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_23497832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?

m
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_2350393

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
:?????????<*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????<2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????<2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????<2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?C
?
"__inference__wrapped_model_2349464
conv2d_8_input8
4sequential_2_conv2d_8_conv2d_readvariableop_resource9
5sequential_2_conv2d_8_biasadd_readvariableop_resource8
4sequential_2_conv2d_9_conv2d_readvariableop_resource9
5sequential_2_conv2d_9_biasadd_readvariableop_resource9
5sequential_2_conv2d_10_conv2d_readvariableop_resource:
6sequential_2_conv2d_10_biasadd_readvariableop_resource9
5sequential_2_conv2d_11_conv2d_readvariableop_resource:
6sequential_2_conv2d_11_biasadd_readvariableop_resource:
6sequential_2_dense_2_mlcmatmul_readvariableop_resource8
4sequential_2_dense_2_biasadd_readvariableop_resource
identity??-sequential_2/conv2d_10/BiasAdd/ReadVariableOp?,sequential_2/conv2d_10/Conv2D/ReadVariableOp?-sequential_2/conv2d_11/BiasAdd/ReadVariableOp?,sequential_2/conv2d_11/Conv2D/ReadVariableOp?,sequential_2/conv2d_8/BiasAdd/ReadVariableOp?+sequential_2/conv2d_8/Conv2D/ReadVariableOp?,sequential_2/conv2d_9/BiasAdd/ReadVariableOp?+sequential_2/conv2d_9/Conv2D/ReadVariableOp?+sequential_2/dense_2/BiasAdd/ReadVariableOp?-sequential_2/dense_2/MLCMatMul/ReadVariableOp?
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_2/conv2d_8/Conv2D/ReadVariableOp?
sequential_2/conv2d_8/Conv2D	MLCConv2Dconv2d_8_input3sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	^*
num_args *
paddingVALID*
strides
2
sequential_2/conv2d_8/Conv2D?
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp?
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	^2
sequential_2/conv2d_8/BiasAdd?
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	^2
sequential_2/conv2d_8/Relu?
+sequential_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02-
+sequential_2/conv2d_9/Conv2D/ReadVariableOp?
sequential_2/conv2d_9/Conv2D	MLCConv2D(sequential_2/conv2d_8/Relu:activations:03sequential_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	<*
num_args *
paddingVALID*
strides
2
sequential_2/conv2d_9/Conv2D?
,sequential_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv2d_9/BiasAdd/ReadVariableOp?
sequential_2/conv2d_9/BiasAddBiasAdd%sequential_2/conv2d_9/Conv2D:output:04sequential_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	<2
sequential_2/conv2d_9/BiasAdd?
sequential_2/conv2d_9/ReluRelu&sequential_2/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	<2
sequential_2/conv2d_9/Relu?
,sequential_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,sequential_2/conv2d_10/Conv2D/ReadVariableOp?
sequential_2/conv2d_10/Conv2D	MLCConv2D(sequential_2/conv2d_9/Relu:activations:04sequential_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
num_args *
paddingVALID*
strides
2
sequential_2/conv2d_10/Conv2D?
-sequential_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_2/conv2d_10/BiasAdd/ReadVariableOp?
sequential_2/conv2d_10/BiasAddBiasAdd&sequential_2/conv2d_10/Conv2D:output:05sequential_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2 
sequential_2/conv2d_10/BiasAdd?
sequential_2/conv2d_10/ReluRelu'sequential_2/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
sequential_2/conv2d_10/Relu?
,sequential_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02.
,sequential_2/conv2d_11/Conv2D/ReadVariableOp?
sequential_2/conv2d_11/Conv2D	MLCConv2D)sequential_2/conv2d_10/Relu:activations:04sequential_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????:
*
num_args *
paddingVALID*
strides
2
sequential_2/conv2d_11/Conv2D?
-sequential_2/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_2/conv2d_11/BiasAdd/ReadVariableOp?
sequential_2/conv2d_11/BiasAddBiasAdd&sequential_2/conv2d_11/Conv2D:output:05sequential_2/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????:
2 
sequential_2/conv2d_11/BiasAdd?
sequential_2/conv2d_11/ReluRelu'sequential_2/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????:
2
sequential_2/conv2d_11/Relu?
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????D  2
sequential_2/flatten_2/Const?
sequential_2/flatten_2/ReshapeReshape)sequential_2/conv2d_11/Relu:activations:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_2/flatten_2/Reshape?
-sequential_2/dense_2/MLCMatMul/ReadVariableOpReadVariableOp6sequential_2_dense_2_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-sequential_2/dense_2/MLCMatMul/ReadVariableOp?
sequential_2/dense_2/MLCMatMul	MLCMatMul'sequential_2/flatten_2/Reshape:output:05sequential_2/dense_2/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_2/dense_2/MLCMatMul?
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_2/BiasAdd/ReadVariableOp?
sequential_2/dense_2/BiasAddBiasAdd(sequential_2/dense_2/MLCMatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_2/BiasAdd?
sequential_2/dense_2/SigmoidSigmoid%sequential_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_2/Sigmoid?
IdentityIdentity sequential_2/dense_2/Sigmoid:y:0.^sequential_2/conv2d_10/BiasAdd/ReadVariableOp-^sequential_2/conv2d_10/Conv2D/ReadVariableOp.^sequential_2/conv2d_11/BiasAdd/ReadVariableOp-^sequential_2/conv2d_11/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp-^sequential_2/conv2d_9/BiasAdd/ReadVariableOp,^sequential_2/conv2d_9/Conv2D/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp.^sequential_2/dense_2/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::2^
-sequential_2/conv2d_10/BiasAdd/ReadVariableOp-sequential_2/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_10/Conv2D/ReadVariableOp,sequential_2/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_11/BiasAdd/ReadVariableOp-sequential_2/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_11/Conv2D/ReadVariableOp,sequential_2/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_9/BiasAdd/ReadVariableOp,sequential_2/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_9/Conv2D/ReadVariableOp+sequential_2/conv2d_9/Conv2D/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2^
-sequential_2/dense_2/MLCMatMul/ReadVariableOp-sequential_2/dense_2/MLCMatMul/ReadVariableOp:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameconv2d_8_input
?

n
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_2350348

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
:?????????	<*
dtype0*
seed???)*
seed2?ϗ2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	<2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	<2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	<2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?

?
D__inference_dense_2_layer_call_and_return_conditional_losses_2350524

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_10_layer_call_fn_2350382

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
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_23496452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	<::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?

o
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_2349700

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
:?????????<*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????<2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????<2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????<2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
.__inference_sequential_2_layer_call_fn_2350197

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
I__inference_sequential_2_layer_call_and_return_conditional_losses_23499212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
l
3__inference_gaussian_noise_10_layer_call_fn_2350402

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
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_23496722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?

m
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_2349755

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
:?????????:
*
dtype0*
seed???)*
seed2շ?2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????:
2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????:
2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????:
2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?

?
D__inference_dense_2_layer_call_and_return_conditional_losses_2349825

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

o
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_2350418

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
:?????????<*
dtype0*
seed???)*
seed2??n2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????<2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????<2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????<2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
m
4__inference_gaussian_dropout_8_layer_call_fn_2350287

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
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_23495342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_2349806

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????D  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?

?
F__inference_conv2d_11_layer_call_and_return_conditional_losses_2349728

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
:?????????:
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
:?????????:
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????:
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_2350282

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2350034
conv2d_8_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
"__inference__wrapped_model_23494642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameconv2d_8_input
?
n
5__inference_gaussian_dropout_10_layer_call_fn_2350427

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
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_23497002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_2350508

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????D  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?

?
E__inference_conv2d_9_layer_call_and_return_conditional_losses_2349562

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
:?????????	<*
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
:?????????	<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????	<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	^::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?
l
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_2349787

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?
m
4__inference_gaussian_dropout_9_layer_call_fn_2350357

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
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_23496172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?
k
2__inference_gaussian_noise_8_layer_call_fn_2350262

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
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_23495062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?
~
)__inference_dense_2_layer_call_fn_2350533

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
D__inference_dense_2_layer_call_and_return_conditional_losses_23498252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_2350327

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?
N
2__inference_gaussian_noise_9_layer_call_fn_2350337

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
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_23495932
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?
O
3__inference_gaussian_noise_10_layer_call_fn_2350407

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
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_23496762
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
l
3__inference_gaussian_noise_11_layer_call_fn_2350472

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
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_23497552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?
j
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_2350467

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_2350257

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?

*__inference_conv2d_9_layer_call_fn_2350312

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
:?????????	<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_23495622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	^::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?

o
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_2350488

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
:?????????:
*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????:
2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????:
2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????:
2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?H
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2349842
conv2d_8_input
conv2d_8_2349490
conv2d_8_2349492
conv2d_9_2349573
conv2d_9_2349575
conv2d_10_2349656
conv2d_10_2349658
conv2d_11_2349739
conv2d_11_2349741
dense_2_2349836
dense_2_2349838
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?+gaussian_dropout_10/StatefulPartitionedCall?+gaussian_dropout_11/StatefulPartitionedCall?*gaussian_dropout_8/StatefulPartitionedCall?*gaussian_dropout_9/StatefulPartitionedCall?)gaussian_noise_10/StatefulPartitionedCall?)gaussian_noise_11/StatefulPartitionedCall?(gaussian_noise_8/StatefulPartitionedCall?(gaussian_noise_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_2349490conv2d_8_2349492*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_23494792"
 conv2d_8/StatefulPartitionedCall?
(gaussian_noise_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_23495062*
(gaussian_noise_8/StatefulPartitionedCall?
*gaussian_dropout_8/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_8/StatefulPartitionedCall:output:0)^gaussian_noise_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_23495342,
*gaussian_dropout_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_8/StatefulPartitionedCall:output:0conv2d_9_2349573conv2d_9_2349575*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_23495622"
 conv2d_9/StatefulPartitionedCall?
(gaussian_noise_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0+^gaussian_dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_23495892*
(gaussian_noise_9/StatefulPartitionedCall?
*gaussian_dropout_9/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_9/StatefulPartitionedCall:output:0)^gaussian_noise_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_23496172,
*gaussian_dropout_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_9/StatefulPartitionedCall:output:0conv2d_10_2349656conv2d_10_2349658*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_23496452#
!conv2d_10/StatefulPartitionedCall?
)gaussian_noise_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0+^gaussian_dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_23496722+
)gaussian_noise_10/StatefulPartitionedCall?
+gaussian_dropout_10/StatefulPartitionedCallStatefulPartitionedCall2gaussian_noise_10/StatefulPartitionedCall:output:0*^gaussian_noise_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_23497002-
+gaussian_dropout_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall4gaussian_dropout_10/StatefulPartitionedCall:output:0conv2d_11_2349739conv2d_11_2349741*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_23497282#
!conv2d_11/StatefulPartitionedCall?
)gaussian_noise_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0,^gaussian_dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_23497552+
)gaussian_noise_11/StatefulPartitionedCall?
+gaussian_dropout_11/StatefulPartitionedCallStatefulPartitionedCall2gaussian_noise_11/StatefulPartitionedCall:output:0*^gaussian_noise_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_23497832-
+gaussian_dropout_11/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall4gaussian_dropout_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_23498062
flatten_2/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_2_2349836dense_2_2349838*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_23498252!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^gaussian_dropout_10/StatefulPartitionedCall,^gaussian_dropout_11/StatefulPartitionedCall+^gaussian_dropout_8/StatefulPartitionedCall+^gaussian_dropout_9/StatefulPartitionedCall*^gaussian_noise_10/StatefulPartitionedCall*^gaussian_noise_11/StatefulPartitionedCall)^gaussian_noise_8/StatefulPartitionedCall)^gaussian_noise_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+gaussian_dropout_10/StatefulPartitionedCall+gaussian_dropout_10/StatefulPartitionedCall2Z
+gaussian_dropout_11/StatefulPartitionedCall+gaussian_dropout_11/StatefulPartitionedCall2X
*gaussian_dropout_8/StatefulPartitionedCall*gaussian_dropout_8/StatefulPartitionedCall2X
*gaussian_dropout_9/StatefulPartitionedCall*gaussian_dropout_9/StatefulPartitionedCall2V
)gaussian_noise_10/StatefulPartitionedCall)gaussian_noise_10/StatefulPartitionedCall2V
)gaussian_noise_11/StatefulPartitionedCall)gaussian_noise_11/StatefulPartitionedCall2T
(gaussian_noise_8/StatefulPartitionedCall(gaussian_noise_8/StatefulPartitionedCall2T
(gaussian_noise_9/StatefulPartitionedCall(gaussian_noise_9/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameconv2d_8_input
?

?
F__inference_conv2d_10_layer_call_and_return_conditional_losses_2349645

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
:?????????<*
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
:?????????<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?H
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2349921

inputs
conv2d_8_2349886
conv2d_8_2349888
conv2d_9_2349893
conv2d_9_2349895
conv2d_10_2349900
conv2d_10_2349902
conv2d_11_2349907
conv2d_11_2349909
dense_2_2349915
dense_2_2349917
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?+gaussian_dropout_10/StatefulPartitionedCall?+gaussian_dropout_11/StatefulPartitionedCall?*gaussian_dropout_8/StatefulPartitionedCall?*gaussian_dropout_9/StatefulPartitionedCall?)gaussian_noise_10/StatefulPartitionedCall?)gaussian_noise_11/StatefulPartitionedCall?(gaussian_noise_8/StatefulPartitionedCall?(gaussian_noise_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_2349886conv2d_8_2349888*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_23494792"
 conv2d_8/StatefulPartitionedCall?
(gaussian_noise_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_23495062*
(gaussian_noise_8/StatefulPartitionedCall?
*gaussian_dropout_8/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_8/StatefulPartitionedCall:output:0)^gaussian_noise_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_23495342,
*gaussian_dropout_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_8/StatefulPartitionedCall:output:0conv2d_9_2349893conv2d_9_2349895*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_23495622"
 conv2d_9/StatefulPartitionedCall?
(gaussian_noise_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0+^gaussian_dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_23495892*
(gaussian_noise_9/StatefulPartitionedCall?
*gaussian_dropout_9/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_9/StatefulPartitionedCall:output:0)^gaussian_noise_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_23496172,
*gaussian_dropout_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_9/StatefulPartitionedCall:output:0conv2d_10_2349900conv2d_10_2349902*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_23496452#
!conv2d_10/StatefulPartitionedCall?
)gaussian_noise_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0+^gaussian_dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_23496722+
)gaussian_noise_10/StatefulPartitionedCall?
+gaussian_dropout_10/StatefulPartitionedCallStatefulPartitionedCall2gaussian_noise_10/StatefulPartitionedCall:output:0*^gaussian_noise_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_23497002-
+gaussian_dropout_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall4gaussian_dropout_10/StatefulPartitionedCall:output:0conv2d_11_2349907conv2d_11_2349909*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_23497282#
!conv2d_11/StatefulPartitionedCall?
)gaussian_noise_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0,^gaussian_dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_23497552+
)gaussian_noise_11/StatefulPartitionedCall?
+gaussian_dropout_11/StatefulPartitionedCallStatefulPartitionedCall2gaussian_noise_11/StatefulPartitionedCall:output:0*^gaussian_noise_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_23497832-
+gaussian_dropout_11/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall4gaussian_dropout_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_23498062
flatten_2/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_2_2349915dense_2_2349917*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_23498252!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^gaussian_dropout_10/StatefulPartitionedCall,^gaussian_dropout_11/StatefulPartitionedCall+^gaussian_dropout_8/StatefulPartitionedCall+^gaussian_dropout_9/StatefulPartitionedCall*^gaussian_noise_10/StatefulPartitionedCall*^gaussian_noise_11/StatefulPartitionedCall)^gaussian_noise_8/StatefulPartitionedCall)^gaussian_noise_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+gaussian_dropout_10/StatefulPartitionedCall+gaussian_dropout_10/StatefulPartitionedCall2Z
+gaussian_dropout_11/StatefulPartitionedCall+gaussian_dropout_11/StatefulPartitionedCall2X
*gaussian_dropout_8/StatefulPartitionedCall*gaussian_dropout_8/StatefulPartitionedCall2X
*gaussian_dropout_9/StatefulPartitionedCall*gaussian_dropout_9/StatefulPartitionedCall2V
)gaussian_noise_10/StatefulPartitionedCall)gaussian_noise_10/StatefulPartitionedCall2V
)gaussian_noise_11/StatefulPartitionedCall)gaussian_noise_11/StatefulPartitionedCall2T
(gaussian_noise_8/StatefulPartitionedCall(gaussian_noise_8/StatefulPartitionedCall2T
(gaussian_noise_9/StatefulPartitionedCall(gaussian_noise_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
??
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2350131

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource-
)dense_2_mlcmatmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/MLCMatMul/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2D	MLCConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	^*
num_args *
paddingVALID*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	^2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	^2
conv2d_8/Relu{
gaussian_noise_8/ShapeShapeconv2d_8/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_8/Shape?
#gaussian_noise_8/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gaussian_noise_8/random_normal/mean?
%gaussian_noise_8/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gaussian_noise_8/random_normal/stddev?
3gaussian_noise_8/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_8/Shape:output:0*
T0*/
_output_shapes
:?????????	^*
dtype0*
seed???)*
seed2???25
3gaussian_noise_8/random_normal/RandomStandardNormal?
"gaussian_noise_8/random_normal/mulMul<gaussian_noise_8/random_normal/RandomStandardNormal:output:0.gaussian_noise_8/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	^2$
"gaussian_noise_8/random_normal/mul?
gaussian_noise_8/random_normalAdd&gaussian_noise_8/random_normal/mul:z:0,gaussian_noise_8/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	^2 
gaussian_noise_8/random_normal?
gaussian_noise_8/addAddV2conv2d_8/Relu:activations:0"gaussian_noise_8/random_normal:z:0*
T0*/
_output_shapes
:?????????	^2
gaussian_noise_8/add|
gaussian_dropout_8/ShapeShapegaussian_noise_8/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_8/Shape?
%gaussian_dropout_8/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gaussian_dropout_8/random_normal/mean?
'gaussian_dropout_8/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2)
'gaussian_dropout_8/random_normal/stddev?
5gaussian_dropout_8/random_normal/RandomStandardNormalRandomStandardNormal!gaussian_dropout_8/Shape:output:0*
T0*/
_output_shapes
:?????????	^*
dtype0*
seed???)*
seed2ѭ?27
5gaussian_dropout_8/random_normal/RandomStandardNormal?
$gaussian_dropout_8/random_normal/mulMul>gaussian_dropout_8/random_normal/RandomStandardNormal:output:00gaussian_dropout_8/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	^2&
$gaussian_dropout_8/random_normal/mul?
 gaussian_dropout_8/random_normalAdd(gaussian_dropout_8/random_normal/mul:z:0.gaussian_dropout_8/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	^2"
 gaussian_dropout_8/random_normal?
gaussian_dropout_8/mulMulgaussian_noise_8/add:z:0$gaussian_dropout_8/random_normal:z:0*
T0*/
_output_shapes
:?????????	^2
gaussian_dropout_8/mul?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2D	MLCConv2Dgaussian_dropout_8/mul:z:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	<*
num_args *
paddingVALID*
strides
2
conv2d_9/Conv2D?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	<2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	<2
conv2d_9/Relu{
gaussian_noise_9/ShapeShapeconv2d_9/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_9/Shape?
#gaussian_noise_9/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gaussian_noise_9/random_normal/mean?
%gaussian_noise_9/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gaussian_noise_9/random_normal/stddev?
3gaussian_noise_9/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_9/Shape:output:0*
T0*/
_output_shapes
:?????????	<*
dtype0*
seed???)*
seed2ױ?25
3gaussian_noise_9/random_normal/RandomStandardNormal?
"gaussian_noise_9/random_normal/mulMul<gaussian_noise_9/random_normal/RandomStandardNormal:output:0.gaussian_noise_9/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	<2$
"gaussian_noise_9/random_normal/mul?
gaussian_noise_9/random_normalAdd&gaussian_noise_9/random_normal/mul:z:0,gaussian_noise_9/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	<2 
gaussian_noise_9/random_normal?
gaussian_noise_9/addAddV2conv2d_9/Relu:activations:0"gaussian_noise_9/random_normal:z:0*
T0*/
_output_shapes
:?????????	<2
gaussian_noise_9/add|
gaussian_dropout_9/ShapeShapegaussian_noise_9/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_9/Shape?
%gaussian_dropout_9/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gaussian_dropout_9/random_normal/mean?
'gaussian_dropout_9/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2)
'gaussian_dropout_9/random_normal/stddev?
5gaussian_dropout_9/random_normal/RandomStandardNormalRandomStandardNormal!gaussian_dropout_9/Shape:output:0*
T0*/
_output_shapes
:?????????	<*
dtype0*
seed???)*
seed2???27
5gaussian_dropout_9/random_normal/RandomStandardNormal?
$gaussian_dropout_9/random_normal/mulMul>gaussian_dropout_9/random_normal/RandomStandardNormal:output:00gaussian_dropout_9/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	<2&
$gaussian_dropout_9/random_normal/mul?
 gaussian_dropout_9/random_normalAdd(gaussian_dropout_9/random_normal/mul:z:0.gaussian_dropout_9/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	<2"
 gaussian_dropout_9/random_normal?
gaussian_dropout_9/mulMulgaussian_noise_9/add:z:0$gaussian_dropout_9/random_normal:z:0*
T0*/
_output_shapes
:?????????	<2
gaussian_dropout_9/mul?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2D	MLCConv2Dgaussian_dropout_9/mul:z:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
num_args *
paddingVALID*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
conv2d_10/Relu~
gaussian_noise_10/ShapeShapeconv2d_10/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_10/Shape?
$gaussian_noise_10/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$gaussian_noise_10/random_normal/mean?
&gaussian_noise_10/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2(
&gaussian_noise_10/random_normal/stddev?
4gaussian_noise_10/random_normal/RandomStandardNormalRandomStandardNormal gaussian_noise_10/Shape:output:0*
T0*/
_output_shapes
:?????????<*
dtype0*
seed???)*
seed2??$26
4gaussian_noise_10/random_normal/RandomStandardNormal?
#gaussian_noise_10/random_normal/mulMul=gaussian_noise_10/random_normal/RandomStandardNormal:output:0/gaussian_noise_10/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????<2%
#gaussian_noise_10/random_normal/mul?
gaussian_noise_10/random_normalAdd'gaussian_noise_10/random_normal/mul:z:0-gaussian_noise_10/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????<2!
gaussian_noise_10/random_normal?
gaussian_noise_10/addAddV2conv2d_10/Relu:activations:0#gaussian_noise_10/random_normal:z:0*
T0*/
_output_shapes
:?????????<2
gaussian_noise_10/add
gaussian_dropout_10/ShapeShapegaussian_noise_10/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_10/Shape?
&gaussian_dropout_10/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&gaussian_dropout_10/random_normal/mean?
(gaussian_dropout_10/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(gaussian_dropout_10/random_normal/stddev?
6gaussian_dropout_10/random_normal/RandomStandardNormalRandomStandardNormal"gaussian_dropout_10/Shape:output:0*
T0*/
_output_shapes
:?????????<*
dtype0*
seed???)*
seed2???28
6gaussian_dropout_10/random_normal/RandomStandardNormal?
%gaussian_dropout_10/random_normal/mulMul?gaussian_dropout_10/random_normal/RandomStandardNormal:output:01gaussian_dropout_10/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????<2'
%gaussian_dropout_10/random_normal/mul?
!gaussian_dropout_10/random_normalAdd)gaussian_dropout_10/random_normal/mul:z:0/gaussian_dropout_10/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????<2#
!gaussian_dropout_10/random_normal?
gaussian_dropout_10/mulMulgaussian_noise_10/add:z:0%gaussian_dropout_10/random_normal:z:0*
T0*/
_output_shapes
:?????????<2
gaussian_dropout_10/mul?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2D	MLCConv2Dgaussian_dropout_10/mul:z:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????:
*
num_args *
paddingVALID*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????:
2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????:
2
conv2d_11/Relu~
gaussian_noise_11/ShapeShapeconv2d_11/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_11/Shape?
$gaussian_noise_11/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$gaussian_noise_11/random_normal/mean?
&gaussian_noise_11/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2(
&gaussian_noise_11/random_normal/stddev?
4gaussian_noise_11/random_normal/RandomStandardNormalRandomStandardNormal gaussian_noise_11/Shape:output:0*
T0*/
_output_shapes
:?????????:
*
dtype0*
seed???)*
seed2???26
4gaussian_noise_11/random_normal/RandomStandardNormal?
#gaussian_noise_11/random_normal/mulMul=gaussian_noise_11/random_normal/RandomStandardNormal:output:0/gaussian_noise_11/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????:
2%
#gaussian_noise_11/random_normal/mul?
gaussian_noise_11/random_normalAdd'gaussian_noise_11/random_normal/mul:z:0-gaussian_noise_11/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????:
2!
gaussian_noise_11/random_normal?
gaussian_noise_11/addAddV2conv2d_11/Relu:activations:0#gaussian_noise_11/random_normal:z:0*
T0*/
_output_shapes
:?????????:
2
gaussian_noise_11/add
gaussian_dropout_11/ShapeShapegaussian_noise_11/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_11/Shape?
&gaussian_dropout_11/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&gaussian_dropout_11/random_normal/mean?
(gaussian_dropout_11/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(gaussian_dropout_11/random_normal/stddev?
6gaussian_dropout_11/random_normal/RandomStandardNormalRandomStandardNormal"gaussian_dropout_11/Shape:output:0*
T0*/
_output_shapes
:?????????:
*
dtype0*
seed???)*
seed2???28
6gaussian_dropout_11/random_normal/RandomStandardNormal?
%gaussian_dropout_11/random_normal/mulMul?gaussian_dropout_11/random_normal/RandomStandardNormal:output:01gaussian_dropout_11/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????:
2'
%gaussian_dropout_11/random_normal/mul?
!gaussian_dropout_11/random_normalAdd)gaussian_dropout_11/random_normal/mul:z:0/gaussian_dropout_11/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????:
2#
!gaussian_dropout_11/random_normal?
gaussian_dropout_11/mulMulgaussian_noise_11/add:z:0%gaussian_dropout_11/random_normal:z:0*
T0*/
_output_shapes
:?????????:
2
gaussian_dropout_11/muls
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????D  2
flatten_2/Const?
flatten_2/ReshapeReshapegaussian_dropout_11/mul:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_2/Reshape?
 dense_2/MLCMatMul/ReadVariableOpReadVariableOp)dense_2_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_2/MLCMatMul/ReadVariableOp?
dense_2/MLCMatMul	MLCMatMulflatten_2/Reshape:output:0(dense_2/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MLCMatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MLCMatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Sigmoid?
IdentityIdentitydense_2/Sigmoid:y:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/MLCMatMul/ReadVariableOp dense_2/MLCMatMul/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?	
?
.__inference_sequential_2_layer_call_fn_2349944
conv2d_8_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
I__inference_sequential_2_layer_call_and_return_conditional_losses_23499212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameconv2d_8_input
?
P
4__inference_gaussian_dropout_8_layer_call_fn_2350292

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
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_23495382
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?

?
E__inference_conv2d_9_layer_call_and_return_conditional_losses_2350303

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
:?????????	<*
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
:?????????	<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????	<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	^::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?

m
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_2349672

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
:?????????<*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????<2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????<2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????<2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
l
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_2349704

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
P
4__inference_gaussian_dropout_9_layer_call_fn_2350362

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
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_23496212
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?9
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2349880
conv2d_8_input
conv2d_8_2349845
conv2d_8_2349847
conv2d_9_2349852
conv2d_9_2349854
conv2d_10_2349859
conv2d_10_2349861
conv2d_11_2349866
conv2d_11_2349868
dense_2_2349874
dense_2_2349876
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_2349845conv2d_8_2349847*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_23494792"
 conv2d_8/StatefulPartitionedCall?
 gaussian_noise_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_23495102"
 gaussian_noise_8/PartitionedCall?
"gaussian_dropout_8/PartitionedCallPartitionedCall)gaussian_noise_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_23495382$
"gaussian_dropout_8/PartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_8/PartitionedCall:output:0conv2d_9_2349852conv2d_9_2349854*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_23495622"
 conv2d_9/StatefulPartitionedCall?
 gaussian_noise_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_23495932"
 gaussian_noise_9/PartitionedCall?
"gaussian_dropout_9/PartitionedCallPartitionedCall)gaussian_noise_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_23496212$
"gaussian_dropout_9/PartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_9/PartitionedCall:output:0conv2d_10_2349859conv2d_10_2349861*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_23496452#
!conv2d_10/StatefulPartitionedCall?
!gaussian_noise_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_23496762#
!gaussian_noise_10/PartitionedCall?
#gaussian_dropout_10/PartitionedCallPartitionedCall*gaussian_noise_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_23497042%
#gaussian_dropout_10/PartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall,gaussian_dropout_10/PartitionedCall:output:0conv2d_11_2349866conv2d_11_2349868*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_23497282#
!conv2d_11/StatefulPartitionedCall?
!gaussian_noise_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_23497592#
!gaussian_noise_11/PartitionedCall?
#gaussian_dropout_11/PartitionedCallPartitionedCall*gaussian_noise_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_23497872%
#gaussian_dropout_11/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall,gaussian_dropout_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_23498062
flatten_2/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_2_2349874dense_2_2349876*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_23498252!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameconv2d_8_input
?
Q
5__inference_gaussian_dropout_10_layer_call_fn_2350432

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
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_23497042
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_2349538

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?

?
E__inference_conv2d_8_layer_call_and_return_conditional_losses_2350233

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
:?????????	^*
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
:?????????	^2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????	^2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?	
?
.__inference_sequential_2_layer_call_fn_2350007
conv2d_8_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
I__inference_sequential_2_layer_call_and_return_conditional_losses_23499842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameconv2d_8_input
?
j
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_2349759

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?
k
2__inference_gaussian_noise_9_layer_call_fn_2350332

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
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_23495892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?

n
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_2350278

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
:?????????	^*
dtype0*
seed???)*
seed2??V2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	^2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	^2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	^2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?9
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2349984

inputs
conv2d_8_2349949
conv2d_8_2349951
conv2d_9_2349956
conv2d_9_2349958
conv2d_10_2349963
conv2d_10_2349965
conv2d_11_2349970
conv2d_11_2349972
dense_2_2349978
dense_2_2349980
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_2349949conv2d_8_2349951*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_23494792"
 conv2d_8/StatefulPartitionedCall?
 gaussian_noise_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_23495102"
 gaussian_noise_8/PartitionedCall?
"gaussian_dropout_8/PartitionedCallPartitionedCall)gaussian_noise_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_23495382$
"gaussian_dropout_8/PartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_8/PartitionedCall:output:0conv2d_9_2349956conv2d_9_2349958*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_23495622"
 conv2d_9/StatefulPartitionedCall?
 gaussian_noise_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_23495932"
 gaussian_noise_9/PartitionedCall?
"gaussian_dropout_9/PartitionedCallPartitionedCall)gaussian_noise_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_23496212$
"gaussian_dropout_9/PartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_9/PartitionedCall:output:0conv2d_10_2349963conv2d_10_2349965*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_23496452#
!conv2d_10/StatefulPartitionedCall?
!gaussian_noise_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_23496762#
!gaussian_noise_10/PartitionedCall?
#gaussian_dropout_10/PartitionedCallPartitionedCall*gaussian_noise_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_23497042%
#gaussian_dropout_10/PartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall,gaussian_dropout_10/PartitionedCall:output:0conv2d_11_2349970conv2d_11_2349972*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_23497282#
!conv2d_11/StatefulPartitionedCall?
!gaussian_noise_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_23497592#
!gaussian_noise_11/PartitionedCall?
#gaussian_dropout_11/PartitionedCallPartitionedCall*gaussian_noise_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_23497872%
#gaussian_dropout_11/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall,gaussian_dropout_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_23498062
flatten_2/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_2_2349978dense_2_2349980*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_23498252!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
j
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_2350397

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?

n
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_2349534

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
:?????????	^*
dtype0*
seed???)*
seed2??2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	^2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	^2
random_normalf
mulMulinputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	^2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?

?
E__inference_conv2d_8_layer_call_and_return_conditional_losses_2349479

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
:?????????	^*
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
:?????????	^2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????	^2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
l
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_2350422

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
+__inference_conv2d_11_layer_call_fn_2350452

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
:?????????:
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_23497282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
j
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_2349676

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
k
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_2350352

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?

m
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_2350463

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
:?????????:
*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????:
2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????:
2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????:
2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?

?
F__inference_conv2d_10_layer_call_and_return_conditional_losses_2350373

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
:?????????<*
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
:?????????<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????	<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_2350776
file_prefix$
 assignvariableop_conv2d_8_kernel$
 assignvariableop_1_conv2d_8_bias&
"assignvariableop_2_conv2d_9_kernel$
 assignvariableop_3_conv2d_9_bias'
#assignvariableop_4_conv2d_10_kernel%
!assignvariableop_5_conv2d_10_bias'
#assignvariableop_6_conv2d_11_kernel%
!assignvariableop_7_conv2d_11_bias%
!assignvariableop_8_dense_2_kernel#
assignvariableop_9_dense_2_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate.
*assignvariableop_15_adam_conv2d_8_kernel_m,
(assignvariableop_16_adam_conv2d_8_bias_m.
*assignvariableop_17_adam_conv2d_9_kernel_m,
(assignvariableop_18_adam_conv2d_9_bias_m/
+assignvariableop_19_adam_conv2d_10_kernel_m-
)assignvariableop_20_adam_conv2d_10_bias_m/
+assignvariableop_21_adam_conv2d_11_kernel_m-
)assignvariableop_22_adam_conv2d_11_bias_m-
)assignvariableop_23_adam_dense_2_kernel_m+
'assignvariableop_24_adam_dense_2_bias_m.
*assignvariableop_25_adam_conv2d_8_kernel_v,
(assignvariableop_26_adam_conv2d_8_bias_v.
*assignvariableop_27_adam_conv2d_9_kernel_v,
(assignvariableop_28_adam_conv2d_9_bias_v/
+assignvariableop_29_adam_conv2d_10_kernel_v-
)assignvariableop_30_adam_conv2d_10_bias_v/
+assignvariableop_31_adam_conv2d_11_kernel_v-
)assignvariableop_32_adam_conv2d_11_bias_v-
)assignvariableop_33_adam_dense_2_kernel_v+
'assignvariableop_34_adam_dense_2_bias_v
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
AssignVariableOpAssignVariableOp assignvariableop_conv2d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_conv2d_8_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_conv2d_8_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv2d_9_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv2d_9_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_10_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_10_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_11_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_11_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_8_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_8_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_9_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_9_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_10_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_10_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_11_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_11_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_2_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_2_bias_vIdentity_34:output:0"/device:CPU:0*
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
?
G
+__inference_flatten_2_layer_call_fn_2350513

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_23498062
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?

?
F__inference_conv2d_11_layer_call_and_return_conditional_losses_2350443

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
:?????????:
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
:?????????:
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????:
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_2349593

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?

l
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_2350253

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
:?????????	^*
dtype0*
seed???)*
seed2?Ȝ2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	^2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	^2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	^2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?

*__inference_conv2d_8_layer_call_fn_2350242

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
:?????????	^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_23494792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
Q
5__inference_gaussian_dropout_11_layer_call_fn_2350502

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
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_23497872
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?N
?
 __inference__traced_save_2350661
file_prefix.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop5
1savev2_adam_conv2d_8_kernel_m_read_readvariableop3
/savev2_adam_conv2d_8_bias_m_read_readvariableop5
1savev2_adam_conv2d_9_kernel_m_read_readvariableop3
/savev2_adam_conv2d_9_bias_m_read_readvariableop6
2savev2_adam_conv2d_10_kernel_m_read_readvariableop4
0savev2_adam_conv2d_10_bias_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_8_kernel_v_read_readvariableop3
/savev2_adam_conv2d_8_bias_v_read_readvariableop5
1savev2_adam_conv2d_9_kernel_v_read_readvariableop3
/savev2_adam_conv2d_9_bias_v_read_readvariableop6
2savev2_adam_conv2d_10_kernel_v_read_readvariableop4
0savev2_adam_conv2d_10_bias_v_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop1savev2_adam_conv2d_8_kernel_m_read_readvariableop/savev2_adam_conv2d_8_bias_m_read_readvariableop1savev2_adam_conv2d_9_kernel_m_read_readvariableop/savev2_adam_conv2d_9_bias_m_read_readvariableop2savev2_adam_conv2d_10_kernel_m_read_readvariableop0savev2_adam_conv2d_10_bias_m_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop1savev2_adam_conv2d_8_kernel_v_read_readvariableop/savev2_adam_conv2d_8_bias_v_read_readvariableop1savev2_adam_conv2d_9_kernel_v_read_readvariableop/savev2_adam_conv2d_9_bias_v_read_readvariableop2savev2_adam_conv2d_10_kernel_v_read_readvariableop0savev2_adam_conv2d_10_bias_v_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
:	?:: : : : : :::#::::
:
:	?::::#::::
:
:	?:: 2(
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
:	?: 
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
:	?: 
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
:	?: #

_output_shapes
::$

_output_shapes
: 
?
O
3__inference_gaussian_noise_11_layer_call_fn_2350477

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
:?????????:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_23497592
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????:
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:
:W S
/
_output_shapes
:?????????:

 
_user_specified_nameinputs
?

l
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_2349506

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
:?????????	^*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	^2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	^2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	^2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?
N
2__inference_gaussian_noise_8_layer_call_fn_2350267

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
:?????????	^* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_23495102
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs
?

l
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_2350323

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
:?????????	<*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	<2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	<2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	<2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?

l
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_2349589

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
:?????????	<*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	<2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	<2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:?????????	<2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????	<2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	<:W S
/
_output_shapes
:?????????	<
 
_user_specified_nameinputs
?
?
.__inference_sequential_2_layer_call_fn_2350222

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
I__inference_sequential_2_layer_call_and_return_conditional_losses_23499842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
i
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_2349510

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	^2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	^:W S
/
_output_shapes
:?????????	^
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
conv2d_8_input?
 serving_default_conv2d_8_input:0?????????`;
dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Ϗ
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?P
_tf_keras_sequential?O{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_8", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [1, 35]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_9", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_10", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_11", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 96, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_8", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [1, 35]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_9", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_10", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_11", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 96, 1]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_8", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
regularization_losses
 	variables
!trainable_variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1}}
?


#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [1, 35]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 94, 30]}}
?
)regularization_losses
*	variables
+trainable_variables
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_9", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
-regularization_losses
.	variables
/trainable_variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1}}
?


1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 60, 30]}}
?
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_10", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
;regularization_losses
<	variables
=trainable_variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1}}
?


?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 96, 1]}, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 60, 30]}}
?
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_11", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5}}
?
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Qkernel
Rbias
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 580}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 580]}}
?
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratem?m?#m?$m?1m?2m??m?@m?Qm?Rm?v?v?#v?$v?1v?2v??v?@v?Qv?Rv?"
	optimizer
 "
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
?
\metrics
regularization_losses
	variables
]layer_regularization_losses
^layer_metrics
_non_trainable_variables

`layers
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):'2conv2d_8/kernel
:2conv2d_8/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
ametrics
regularization_losses
	variables
blayer_regularization_losses
clayer_metrics
dnon_trainable_variables
trainable_variables

elayers
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
fmetrics
regularization_losses
	variables
glayer_regularization_losses
hlayer_metrics
inon_trainable_variables
trainable_variables

jlayers
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
kmetrics
regularization_losses
 	variables
llayer_regularization_losses
mlayer_metrics
nnon_trainable_variables
!trainable_variables

olayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'#2conv2d_9/kernel
:2conv2d_9/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
pmetrics
%regularization_losses
&	variables
qlayer_regularization_losses
rlayer_metrics
snon_trainable_variables
'trainable_variables

tlayers
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
umetrics
)regularization_losses
*	variables
vlayer_regularization_losses
wlayer_metrics
xnon_trainable_variables
+trainable_variables

ylayers
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
zmetrics
-regularization_losses
.	variables
{layer_regularization_losses
|layer_metrics
}non_trainable_variables
/trainable_variables

~layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_10/kernel
:2conv2d_10/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
metrics
3regularization_losses
4	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
5trainable_variables
?layers
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
?metrics
7regularization_losses
8	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
9trainable_variables
?layers
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
?metrics
;regularization_losses
<	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
=trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
2conv2d_11/kernel
:
2conv2d_11/bias
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
?metrics
Aregularization_losses
B	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Ctrainable_variables
?layers
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
?metrics
Eregularization_losses
F	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Gtrainable_variables
?layers
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
?metrics
Iregularization_losses
J	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Ktrainable_variables
?layers
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
?metrics
Mregularization_losses
N	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Otrainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
?
?metrics
Sregularization_losses
T	variables
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
Utrainable_variables
?layers
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
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.:,2Adam/conv2d_8/kernel/m
 :2Adam/conv2d_8/bias/m
.:,#2Adam/conv2d_9/kernel/m
 :2Adam/conv2d_9/bias/m
/:-2Adam/conv2d_10/kernel/m
!:2Adam/conv2d_10/bias/m
/:-
2Adam/conv2d_11/kernel/m
!:
2Adam/conv2d_11/bias/m
&:$	?2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
.:,2Adam/conv2d_8/kernel/v
 :2Adam/conv2d_8/bias/v
.:,#2Adam/conv2d_9/kernel/v
 :2Adam/conv2d_9/bias/v
/:-2Adam/conv2d_10/kernel/v
!:2Adam/conv2d_10/bias/v
/:-
2Adam/conv2d_11/kernel/v
!:
2Adam/conv2d_11/bias/v
&:$	?2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
?2?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2349842
I__inference_sequential_2_layer_call_and_return_conditional_losses_2349880
I__inference_sequential_2_layer_call_and_return_conditional_losses_2350172
I__inference_sequential_2_layer_call_and_return_conditional_losses_2350131?
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
"__inference__wrapped_model_2349464?
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
conv2d_8_input?????????`
?2?
.__inference_sequential_2_layer_call_fn_2350222
.__inference_sequential_2_layer_call_fn_2350197
.__inference_sequential_2_layer_call_fn_2350007
.__inference_sequential_2_layer_call_fn_2349944?
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
E__inference_conv2d_8_layer_call_and_return_conditional_losses_2350233?
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
*__inference_conv2d_8_layer_call_fn_2350242?
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
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_2350257
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_2350253?
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
2__inference_gaussian_noise_8_layer_call_fn_2350267
2__inference_gaussian_noise_8_layer_call_fn_2350262?
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
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_2350282
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_2350278?
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
4__inference_gaussian_dropout_8_layer_call_fn_2350287
4__inference_gaussian_dropout_8_layer_call_fn_2350292?
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
E__inference_conv2d_9_layer_call_and_return_conditional_losses_2350303?
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
*__inference_conv2d_9_layer_call_fn_2350312?
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
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_2350327
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_2350323?
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
2__inference_gaussian_noise_9_layer_call_fn_2350332
2__inference_gaussian_noise_9_layer_call_fn_2350337?
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
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_2350352
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_2350348?
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
4__inference_gaussian_dropout_9_layer_call_fn_2350357
4__inference_gaussian_dropout_9_layer_call_fn_2350362?
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
F__inference_conv2d_10_layer_call_and_return_conditional_losses_2350373?
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
+__inference_conv2d_10_layer_call_fn_2350382?
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
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_2350397
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_2350393?
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
3__inference_gaussian_noise_10_layer_call_fn_2350407
3__inference_gaussian_noise_10_layer_call_fn_2350402?
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
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_2350422
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_2350418?
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
5__inference_gaussian_dropout_10_layer_call_fn_2350427
5__inference_gaussian_dropout_10_layer_call_fn_2350432?
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
F__inference_conv2d_11_layer_call_and_return_conditional_losses_2350443?
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
+__inference_conv2d_11_layer_call_fn_2350452?
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
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_2350463
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_2350467?
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
3__inference_gaussian_noise_11_layer_call_fn_2350472
3__inference_gaussian_noise_11_layer_call_fn_2350477?
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
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_2350488
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_2350492?
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
5__inference_gaussian_dropout_11_layer_call_fn_2350502
5__inference_gaussian_dropout_11_layer_call_fn_2350497?
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
F__inference_flatten_2_layer_call_and_return_conditional_losses_2350508?
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
+__inference_flatten_2_layer_call_fn_2350513?
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
D__inference_dense_2_layer_call_and_return_conditional_losses_2350524?
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
)__inference_dense_2_layer_call_fn_2350533?
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
%__inference_signature_wrapper_2350034conv2d_8_input"?
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
"__inference__wrapped_model_2349464?
#$12?@QR??<
5?2
0?-
conv2d_8_input?????????`
? "1?.
,
dense_2!?
dense_2??????????
F__inference_conv2d_10_layer_call_and_return_conditional_losses_2350373l127?4
-?*
(?%
inputs?????????	<
? "-?*
#? 
0?????????<
? ?
+__inference_conv2d_10_layer_call_fn_2350382_127?4
-?*
(?%
inputs?????????	<
? " ??????????<?
F__inference_conv2d_11_layer_call_and_return_conditional_losses_2350443l?@7?4
-?*
(?%
inputs?????????<
? "-?*
#? 
0?????????:

? ?
+__inference_conv2d_11_layer_call_fn_2350452_?@7?4
-?*
(?%
inputs?????????<
? " ??????????:
?
E__inference_conv2d_8_layer_call_and_return_conditional_losses_2350233l7?4
-?*
(?%
inputs?????????`
? "-?*
#? 
0?????????	^
? ?
*__inference_conv2d_8_layer_call_fn_2350242_7?4
-?*
(?%
inputs?????????`
? " ??????????	^?
E__inference_conv2d_9_layer_call_and_return_conditional_losses_2350303l#$7?4
-?*
(?%
inputs?????????	^
? "-?*
#? 
0?????????	<
? ?
*__inference_conv2d_9_layer_call_fn_2350312_#$7?4
-?*
(?%
inputs?????????	^
? " ??????????	<?
D__inference_dense_2_layer_call_and_return_conditional_losses_2350524]QR0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_2_layer_call_fn_2350533PQR0?-
&?#
!?
inputs??????????
? "???????????
F__inference_flatten_2_layer_call_and_return_conditional_losses_2350508a7?4
-?*
(?%
inputs?????????:

? "&?#
?
0??????????
? ?
+__inference_flatten_2_layer_call_fn_2350513T7?4
-?*
(?%
inputs?????????:

? "????????????
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_2350418l;?8
1?.
(?%
inputs?????????<
p
? "-?*
#? 
0?????????<
? ?
P__inference_gaussian_dropout_10_layer_call_and_return_conditional_losses_2350422l;?8
1?.
(?%
inputs?????????<
p 
? "-?*
#? 
0?????????<
? ?
5__inference_gaussian_dropout_10_layer_call_fn_2350427_;?8
1?.
(?%
inputs?????????<
p
? " ??????????<?
5__inference_gaussian_dropout_10_layer_call_fn_2350432_;?8
1?.
(?%
inputs?????????<
p 
? " ??????????<?
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_2350488l;?8
1?.
(?%
inputs?????????:

p
? "-?*
#? 
0?????????:

? ?
P__inference_gaussian_dropout_11_layer_call_and_return_conditional_losses_2350492l;?8
1?.
(?%
inputs?????????:

p 
? "-?*
#? 
0?????????:

? ?
5__inference_gaussian_dropout_11_layer_call_fn_2350497_;?8
1?.
(?%
inputs?????????:

p
? " ??????????:
?
5__inference_gaussian_dropout_11_layer_call_fn_2350502_;?8
1?.
(?%
inputs?????????:

p 
? " ??????????:
?
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_2350278l;?8
1?.
(?%
inputs?????????	^
p
? "-?*
#? 
0?????????	^
? ?
O__inference_gaussian_dropout_8_layer_call_and_return_conditional_losses_2350282l;?8
1?.
(?%
inputs?????????	^
p 
? "-?*
#? 
0?????????	^
? ?
4__inference_gaussian_dropout_8_layer_call_fn_2350287_;?8
1?.
(?%
inputs?????????	^
p
? " ??????????	^?
4__inference_gaussian_dropout_8_layer_call_fn_2350292_;?8
1?.
(?%
inputs?????????	^
p 
? " ??????????	^?
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_2350348l;?8
1?.
(?%
inputs?????????	<
p
? "-?*
#? 
0?????????	<
? ?
O__inference_gaussian_dropout_9_layer_call_and_return_conditional_losses_2350352l;?8
1?.
(?%
inputs?????????	<
p 
? "-?*
#? 
0?????????	<
? ?
4__inference_gaussian_dropout_9_layer_call_fn_2350357_;?8
1?.
(?%
inputs?????????	<
p
? " ??????????	<?
4__inference_gaussian_dropout_9_layer_call_fn_2350362_;?8
1?.
(?%
inputs?????????	<
p 
? " ??????????	<?
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_2350393l;?8
1?.
(?%
inputs?????????<
p
? "-?*
#? 
0?????????<
? ?
N__inference_gaussian_noise_10_layer_call_and_return_conditional_losses_2350397l;?8
1?.
(?%
inputs?????????<
p 
? "-?*
#? 
0?????????<
? ?
3__inference_gaussian_noise_10_layer_call_fn_2350402_;?8
1?.
(?%
inputs?????????<
p
? " ??????????<?
3__inference_gaussian_noise_10_layer_call_fn_2350407_;?8
1?.
(?%
inputs?????????<
p 
? " ??????????<?
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_2350463l;?8
1?.
(?%
inputs?????????:

p
? "-?*
#? 
0?????????:

? ?
N__inference_gaussian_noise_11_layer_call_and_return_conditional_losses_2350467l;?8
1?.
(?%
inputs?????????:

p 
? "-?*
#? 
0?????????:

? ?
3__inference_gaussian_noise_11_layer_call_fn_2350472_;?8
1?.
(?%
inputs?????????:

p
? " ??????????:
?
3__inference_gaussian_noise_11_layer_call_fn_2350477_;?8
1?.
(?%
inputs?????????:

p 
? " ??????????:
?
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_2350253l;?8
1?.
(?%
inputs?????????	^
p
? "-?*
#? 
0?????????	^
? ?
M__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_2350257l;?8
1?.
(?%
inputs?????????	^
p 
? "-?*
#? 
0?????????	^
? ?
2__inference_gaussian_noise_8_layer_call_fn_2350262_;?8
1?.
(?%
inputs?????????	^
p
? " ??????????	^?
2__inference_gaussian_noise_8_layer_call_fn_2350267_;?8
1?.
(?%
inputs?????????	^
p 
? " ??????????	^?
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_2350323l;?8
1?.
(?%
inputs?????????	<
p
? "-?*
#? 
0?????????	<
? ?
M__inference_gaussian_noise_9_layer_call_and_return_conditional_losses_2350327l;?8
1?.
(?%
inputs?????????	<
p 
? "-?*
#? 
0?????????	<
? ?
2__inference_gaussian_noise_9_layer_call_fn_2350332_;?8
1?.
(?%
inputs?????????	<
p
? " ??????????	<?
2__inference_gaussian_noise_9_layer_call_fn_2350337_;?8
1?.
(?%
inputs?????????	<
p 
? " ??????????	<?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2349842|
#$12?@QRG?D
=?:
0?-
conv2d_8_input?????????`
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2349880|
#$12?@QRG?D
=?:
0?-
conv2d_8_input?????????`
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2350131t
#$12?@QR??<
5?2
(?%
inputs?????????`
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_2_layer_call_and_return_conditional_losses_2350172t
#$12?@QR??<
5?2
(?%
inputs?????????`
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_2_layer_call_fn_2349944o
#$12?@QRG?D
=?:
0?-
conv2d_8_input?????????`
p

 
? "???????????
.__inference_sequential_2_layer_call_fn_2350007o
#$12?@QRG?D
=?:
0?-
conv2d_8_input?????????`
p 

 
? "???????????
.__inference_sequential_2_layer_call_fn_2350197g
#$12?@QR??<
5?2
(?%
inputs?????????`
p

 
? "???????????
.__inference_sequential_2_layer_call_fn_2350222g
#$12?@QR??<
5?2
(?%
inputs?????????`
p 

 
? "???????????
%__inference_signature_wrapper_2350034?
#$12?@QRQ?N
? 
G?D
B
conv2d_8_input0?-
conv2d_8_input?????????`"1?.
,
dense_2!?
dense_2?????????