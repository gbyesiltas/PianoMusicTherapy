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
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:#* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:#*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:
*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:
*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
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
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:#*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv2d_3/kernel/m
?
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:#*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv2d_3/kernel/v
?
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?G
value?GB?G B?F
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
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
 trainable_variables
!	variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
R
)regularization_losses
*trainable_variables
+	variables
,	keras_api
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
R
7regularization_losses
8trainable_variables
9	variables
:	keras_api
R
;regularization_losses
<trainable_variables
=	variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
R
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
R
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
R
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
h

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
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
regularization_losses

\layers
trainable_variables
]non_trainable_variables
	variables
^metrics
_layer_regularization_losses
`layer_metrics
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

alayers
trainable_variables
bnon_trainable_variables
	variables
cmetrics
dlayer_regularization_losses
elayer_metrics
 
 
 
?
regularization_losses

flayers
trainable_variables
gnon_trainable_variables
	variables
hmetrics
ilayer_regularization_losses
jlayer_metrics
 
 
 
?
regularization_losses

klayers
 trainable_variables
lnon_trainable_variables
!	variables
mmetrics
nlayer_regularization_losses
olayer_metrics
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
?
%regularization_losses

players
&trainable_variables
qnon_trainable_variables
'	variables
rmetrics
slayer_regularization_losses
tlayer_metrics
 
 
 
?
)regularization_losses

ulayers
*trainable_variables
vnon_trainable_variables
+	variables
wmetrics
xlayer_regularization_losses
ylayer_metrics
 
 
 
?
-regularization_losses

zlayers
.trainable_variables
{non_trainable_variables
/	variables
|metrics
}layer_regularization_losses
~layer_metrics
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
?
3regularization_losses

layers
4trainable_variables
?non_trainable_variables
5	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
7regularization_losses
?layers
8trainable_variables
?non_trainable_variables
9	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
;regularization_losses
?layers
<trainable_variables
?non_trainable_variables
=	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
?
Aregularization_losses
?layers
Btrainable_variables
?non_trainable_variables
C	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
Eregularization_losses
?layers
Ftrainable_variables
?non_trainable_variables
G	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
Iregularization_losses
?layers
Jtrainable_variables
?non_trainable_variables
K	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
Mregularization_losses
?layers
Ntrainable_variables
?non_trainable_variables
O	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
?
Sregularization_losses
?layers
Ttrainable_variables
?non_trainable_variables
U	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
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
 
 
 
 
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_inputPlaceholder*/
_output_shapes
:?????????T*
dtype0*$
shape:?????????T
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/bias*
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
GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_979024
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*0
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
GPU 2J 8? *(
f#R!
__inference__traced_save_979651
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/dense/kernel/vAdam/dense/bias/v*/
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_979766??	
?
f
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_978500

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

?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_978635

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
?

k
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_978662

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
?

m
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_978773

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
seed2?Ż2$
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
?
j
1__inference_gaussian_noise_3_layer_call_fn_979462

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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_9787452
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
A__inference_dense_layer_call_and_return_conditional_losses_978815

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

k
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_979313

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
seed2???2$
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

?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_979293

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
?8
?
F__inference_sequential_layer_call_and_return_conditional_losses_978870
conv2d_input
conv2d_978835
conv2d_978837
conv2d_1_978842
conv2d_1_978844
conv2d_2_978849
conv2d_2_978851
conv2d_3_978856
conv2d_3_978858
dense_978864
dense_978866
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_978835conv2d_978837*
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
GPU 2J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_9784692 
conv2d/StatefulPartitionedCall?
gaussian_noise/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_9785002 
gaussian_noise/PartitionedCall?
 gaussian_dropout/PartitionedCallPartitionedCall'gaussian_noise/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_9785282"
 gaussian_dropout/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)gaussian_dropout/PartitionedCall:output:0conv2d_1_978842conv2d_1_978844*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_9785522"
 conv2d_1/StatefulPartitionedCall?
 gaussian_noise_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_9785832"
 gaussian_noise_1/PartitionedCall?
"gaussian_dropout_1/PartitionedCallPartitionedCall)gaussian_noise_1/PartitionedCall:output:0*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_9786112$
"gaussian_dropout_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_1/PartitionedCall:output:0conv2d_2_978849conv2d_2_978851*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_9786352"
 conv2d_2/StatefulPartitionedCall?
 gaussian_noise_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_9786662"
 gaussian_noise_2/PartitionedCall?
"gaussian_dropout_2/PartitionedCallPartitionedCall)gaussian_noise_2/PartitionedCall:output:0*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_9786942$
"gaussian_dropout_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_2/PartitionedCall:output:0conv2d_3_978856conv2d_3_978858*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_9787182"
 conv2d_3/StatefulPartitionedCall?
 gaussian_noise_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_9787492"
 gaussian_noise_3/PartitionedCall?
"gaussian_dropout_3/PartitionedCallPartitionedCall)gaussian_noise_3/PartitionedCall:output:0*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_9787772$
"gaussian_dropout_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall+gaussian_dropout_3/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_9787962
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_978864dense_978866*
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
GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9788152
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????T
&
_user_specified_nameconv2d_input
?
M
1__inference_gaussian_noise_2_layer_call_fn_979397

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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_9786662
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
?

m
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_979478

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
seed2???2$
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
?4
?
F__inference_sequential_layer_call_and_return_conditional_losses_979162

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'dense_mlcmatmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MLCMatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D	MLCConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R*
num_args *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	R2
conv2d/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D	MLCConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	0*
num_args *
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	02
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	02
conv2d_1/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2D	MLCConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
num_args *
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
conv2d_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D	MLCConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
*
num_args *
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????.
2
conv2d_3/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeconv2d_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MLCMatMul/ReadVariableOpReadVariableOp'dense_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense/MLCMatMul/ReadVariableOp?
dense/MLCMatMul	MLCMatMulflatten/Reshape:output:0&dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MLCMatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MLCMatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Sigmoid?
IdentityIdentitydense/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/MLCMatMul/ReadVariableOpdense/MLCMatMul/ReadVariableOp:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
h
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_978666

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
h
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_978749

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
?
K
/__inference_gaussian_noise_layer_call_fn_979257

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
GPU 2J 8? *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_9785002
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
?
?
$__inference_signature_wrapper_979024
conv2d_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU 2J 8? **
f%R#
!__inference__wrapped_model_9784542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????T
&
_user_specified_nameconv2d_input
?

?
B__inference_conv2d_layer_call_and_return_conditional_losses_978469

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
?
h
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_978528

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
~
)__inference_conv2d_2_layer_call_fn_979372

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_9786352
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
?

m
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_979338

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
seed2??12$
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

k
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_979453

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
seed???)*
seed2ؖ42$
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
?
O
3__inference_gaussian_dropout_1_layer_call_fn_979352

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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_9786112
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
?
l
3__inference_gaussian_dropout_1_layer_call_fn_979347

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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_9786072
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_979363

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
?
~
)__inference_conv2d_3_layer_call_fn_979442

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_9787182
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
?

?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_979433

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
?
j
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_979482

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
?
j
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_978611

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

i
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_978496

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
??
?
F__inference_sequential_layer_call_and_return_conditional_losses_979121

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'dense_mlcmatmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MLCMatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D	MLCConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R*
num_args *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	R2
conv2d/Reluu
gaussian_noise/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise/Shape?
!gaussian_noise/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!gaussian_noise/random_normal/mean?
#gaussian_noise/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2%
#gaussian_noise/random_normal/stddev?
1gaussian_noise/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise/Shape:output:0*
T0*/
_output_shapes
:?????????	R*
dtype0*
seed???)*
seed2???23
1gaussian_noise/random_normal/RandomStandardNormal?
 gaussian_noise/random_normal/mulMul:gaussian_noise/random_normal/RandomStandardNormal:output:0,gaussian_noise/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	R2"
 gaussian_noise/random_normal/mul?
gaussian_noise/random_normalAdd$gaussian_noise/random_normal/mul:z:0*gaussian_noise/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	R2
gaussian_noise/random_normal?
gaussian_noise/addAddV2conv2d/Relu:activations:0 gaussian_noise/random_normal:z:0*
T0*/
_output_shapes
:?????????	R2
gaussian_noise/addv
gaussian_dropout/ShapeShapegaussian_noise/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout/Shape?
#gaussian_dropout/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#gaussian_dropout/random_normal/mean?
%gaussian_dropout/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2'
%gaussian_dropout/random_normal/stddev?
3gaussian_dropout/random_normal/RandomStandardNormalRandomStandardNormalgaussian_dropout/Shape:output:0*
T0*/
_output_shapes
:?????????	R*
dtype0*
seed???)*
seed2???25
3gaussian_dropout/random_normal/RandomStandardNormal?
"gaussian_dropout/random_normal/mulMul<gaussian_dropout/random_normal/RandomStandardNormal:output:0.gaussian_dropout/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	R2$
"gaussian_dropout/random_normal/mul?
gaussian_dropout/random_normalAdd&gaussian_dropout/random_normal/mul:z:0,gaussian_dropout/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	R2 
gaussian_dropout/random_normal?
gaussian_dropout/mulMulgaussian_noise/add:z:0"gaussian_dropout/random_normal:z:0*
T0*/
_output_shapes
:?????????	R2
gaussian_dropout/mul?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D	MLCConv2Dgaussian_dropout/mul:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	0*
num_args *
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	02
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	02
conv2d_1/Relu{
gaussian_noise_1/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_1/Shape?
#gaussian_noise_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gaussian_noise_1/random_normal/mean?
%gaussian_noise_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gaussian_noise_1/random_normal/stddev?
3gaussian_noise_1/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_1/Shape:output:0*
T0*/
_output_shapes
:?????????	0*
dtype0*
seed???)*
seed2??c25
3gaussian_noise_1/random_normal/RandomStandardNormal?
"gaussian_noise_1/random_normal/mulMul<gaussian_noise_1/random_normal/RandomStandardNormal:output:0.gaussian_noise_1/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	02$
"gaussian_noise_1/random_normal/mul?
gaussian_noise_1/random_normalAdd&gaussian_noise_1/random_normal/mul:z:0,gaussian_noise_1/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	02 
gaussian_noise_1/random_normal?
gaussian_noise_1/addAddV2conv2d_1/Relu:activations:0"gaussian_noise_1/random_normal:z:0*
T0*/
_output_shapes
:?????????	02
gaussian_noise_1/add|
gaussian_dropout_1/ShapeShapegaussian_noise_1/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_1/Shape?
%gaussian_dropout_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gaussian_dropout_1/random_normal/mean?
'gaussian_dropout_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2)
'gaussian_dropout_1/random_normal/stddev?
5gaussian_dropout_1/random_normal/RandomStandardNormalRandomStandardNormal!gaussian_dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????	0*
dtype0*
seed???)*
seed2???27
5gaussian_dropout_1/random_normal/RandomStandardNormal?
$gaussian_dropout_1/random_normal/mulMul>gaussian_dropout_1/random_normal/RandomStandardNormal:output:00gaussian_dropout_1/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????	02&
$gaussian_dropout_1/random_normal/mul?
 gaussian_dropout_1/random_normalAdd(gaussian_dropout_1/random_normal/mul:z:0.gaussian_dropout_1/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????	02"
 gaussian_dropout_1/random_normal?
gaussian_dropout_1/mulMulgaussian_noise_1/add:z:0$gaussian_dropout_1/random_normal:z:0*
T0*/
_output_shapes
:?????????	02
gaussian_dropout_1/mul?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2D	MLCConv2Dgaussian_dropout_1/mul:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
num_args *
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
conv2d_2/Relu{
gaussian_noise_2/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_2/Shape?
#gaussian_noise_2/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gaussian_noise_2/random_normal/mean?
%gaussian_noise_2/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gaussian_noise_2/random_normal/stddev?
3gaussian_noise_2/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_2/Shape:output:0*
T0*/
_output_shapes
:?????????0*
dtype0*
seed???)*
seed2胎25
3gaussian_noise_2/random_normal/RandomStandardNormal?
"gaussian_noise_2/random_normal/mulMul<gaussian_noise_2/random_normal/RandomStandardNormal:output:0.gaussian_noise_2/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????02$
"gaussian_noise_2/random_normal/mul?
gaussian_noise_2/random_normalAdd&gaussian_noise_2/random_normal/mul:z:0,gaussian_noise_2/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????02 
gaussian_noise_2/random_normal?
gaussian_noise_2/addAddV2conv2d_2/Relu:activations:0"gaussian_noise_2/random_normal:z:0*
T0*/
_output_shapes
:?????????02
gaussian_noise_2/add|
gaussian_dropout_2/ShapeShapegaussian_noise_2/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_2/Shape?
%gaussian_dropout_2/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gaussian_dropout_2/random_normal/mean?
'gaussian_dropout_2/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???>2)
'gaussian_dropout_2/random_normal/stddev?
5gaussian_dropout_2/random_normal/RandomStandardNormalRandomStandardNormal!gaussian_dropout_2/Shape:output:0*
T0*/
_output_shapes
:?????????0*
dtype0*
seed???)*
seed2??427
5gaussian_dropout_2/random_normal/RandomStandardNormal?
$gaussian_dropout_2/random_normal/mulMul>gaussian_dropout_2/random_normal/RandomStandardNormal:output:00gaussian_dropout_2/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????02&
$gaussian_dropout_2/random_normal/mul?
 gaussian_dropout_2/random_normalAdd(gaussian_dropout_2/random_normal/mul:z:0.gaussian_dropout_2/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????02"
 gaussian_dropout_2/random_normal?
gaussian_dropout_2/mulMulgaussian_noise_2/add:z:0$gaussian_dropout_2/random_normal:z:0*
T0*/
_output_shapes
:?????????02
gaussian_dropout_2/mul?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D	MLCConv2Dgaussian_dropout_2/mul:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
*
num_args *
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????.
2
conv2d_3/Relu{
gaussian_noise_3/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
gaussian_noise_3/Shape?
#gaussian_noise_3/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gaussian_noise_3/random_normal/mean?
%gaussian_noise_3/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gaussian_noise_3/random_normal/stddev?
3gaussian_noise_3/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_3/Shape:output:0*
T0*/
_output_shapes
:?????????.
*
dtype0*
seed???)*
seed2ɔ?25
3gaussian_noise_3/random_normal/RandomStandardNormal?
"gaussian_noise_3/random_normal/mulMul<gaussian_noise_3/random_normal/RandomStandardNormal:output:0.gaussian_noise_3/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????.
2$
"gaussian_noise_3/random_normal/mul?
gaussian_noise_3/random_normalAdd&gaussian_noise_3/random_normal/mul:z:0,gaussian_noise_3/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????.
2 
gaussian_noise_3/random_normal?
gaussian_noise_3/addAddV2conv2d_3/Relu:activations:0"gaussian_noise_3/random_normal:z:0*
T0*/
_output_shapes
:?????????.
2
gaussian_noise_3/add|
gaussian_dropout_3/ShapeShapegaussian_noise_3/add:z:0*
T0*
_output_shapes
:2
gaussian_dropout_3/Shape?
%gaussian_dropout_3/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%gaussian_dropout_3/random_normal/mean?
'gaussian_dropout_3/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'gaussian_dropout_3/random_normal/stddev?
5gaussian_dropout_3/random_normal/RandomStandardNormalRandomStandardNormal!gaussian_dropout_3/Shape:output:0*
T0*/
_output_shapes
:?????????.
*
dtype0*
seed???)*
seed2??l27
5gaussian_dropout_3/random_normal/RandomStandardNormal?
$gaussian_dropout_3/random_normal/mulMul>gaussian_dropout_3/random_normal/RandomStandardNormal:output:00gaussian_dropout_3/random_normal/stddev:output:0*
T0*/
_output_shapes
:?????????.
2&
$gaussian_dropout_3/random_normal/mul?
 gaussian_dropout_3/random_normalAdd(gaussian_dropout_3/random_normal/mul:z:0.gaussian_dropout_3/random_normal/mean:output:0*
T0*/
_output_shapes
:?????????.
2"
 gaussian_dropout_3/random_normal?
gaussian_dropout_3/mulMulgaussian_noise_3/add:z:0$gaussian_dropout_3/random_normal:z:0*
T0*/
_output_shapes
:?????????.
2
gaussian_dropout_3/mulo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapegaussian_dropout_3/mul:z:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MLCMatMul/ReadVariableOpReadVariableOp'dense_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense/MLCMatMul/ReadVariableOp?
dense/MLCMatMul	MLCMatMulflatten/Reshape:output:0&dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MLCMatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MLCMatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Sigmoid?
IdentityIdentitydense/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/MLCMatMul/ReadVariableOpdense/MLCMatMul/ReadVariableOp:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?G
?
F__inference_sequential_layer_call_and_return_conditional_losses_978911

inputs
conv2d_978876
conv2d_978878
conv2d_1_978883
conv2d_1_978885
conv2d_2_978890
conv2d_2_978892
conv2d_3_978897
conv2d_3_978899
dense_978905
dense_978907
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?(gaussian_dropout/StatefulPartitionedCall?*gaussian_dropout_1/StatefulPartitionedCall?*gaussian_dropout_2/StatefulPartitionedCall?*gaussian_dropout_3/StatefulPartitionedCall?&gaussian_noise/StatefulPartitionedCall?(gaussian_noise_1/StatefulPartitionedCall?(gaussian_noise_2/StatefulPartitionedCall?(gaussian_noise_3/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_978876conv2d_978878*
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
GPU 2J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_9784692 
conv2d/StatefulPartitionedCall?
&gaussian_noise/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_9784962(
&gaussian_noise/StatefulPartitionedCall?
(gaussian_dropout/StatefulPartitionedCallStatefulPartitionedCall/gaussian_noise/StatefulPartitionedCall:output:0'^gaussian_noise/StatefulPartitionedCall*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_9785242*
(gaussian_dropout/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1gaussian_dropout/StatefulPartitionedCall:output:0conv2d_1_978883conv2d_1_978885*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_9785522"
 conv2d_1/StatefulPartitionedCall?
(gaussian_noise_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0)^gaussian_dropout/StatefulPartitionedCall*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_9785792*
(gaussian_noise_1/StatefulPartitionedCall?
*gaussian_dropout_1/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_1/StatefulPartitionedCall:output:0)^gaussian_noise_1/StatefulPartitionedCall*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_9786072,
*gaussian_dropout_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_1/StatefulPartitionedCall:output:0conv2d_2_978890conv2d_2_978892*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_9786352"
 conv2d_2/StatefulPartitionedCall?
(gaussian_noise_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0+^gaussian_dropout_1/StatefulPartitionedCall*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_9786622*
(gaussian_noise_2/StatefulPartitionedCall?
*gaussian_dropout_2/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_2/StatefulPartitionedCall:output:0)^gaussian_noise_2/StatefulPartitionedCall*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_9786902,
*gaussian_dropout_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_2/StatefulPartitionedCall:output:0conv2d_3_978897conv2d_3_978899*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_9787182"
 conv2d_3/StatefulPartitionedCall?
(gaussian_noise_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0+^gaussian_dropout_2/StatefulPartitionedCall*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_9787452*
(gaussian_noise_3/StatefulPartitionedCall?
*gaussian_dropout_3/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_3/StatefulPartitionedCall:output:0)^gaussian_noise_3/StatefulPartitionedCall*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_9787732,
*gaussian_dropout_3/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall3gaussian_dropout_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_9787962
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_978905dense_978907*
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
GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9788152
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall)^gaussian_dropout/StatefulPartitionedCall+^gaussian_dropout_1/StatefulPartitionedCall+^gaussian_dropout_2/StatefulPartitionedCall+^gaussian_dropout_3/StatefulPartitionedCall'^gaussian_noise/StatefulPartitionedCall)^gaussian_noise_1/StatefulPartitionedCall)^gaussian_noise_2/StatefulPartitionedCall)^gaussian_noise_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(gaussian_dropout/StatefulPartitionedCall(gaussian_dropout/StatefulPartitionedCall2X
*gaussian_dropout_1/StatefulPartitionedCall*gaussian_dropout_1/StatefulPartitionedCall2X
*gaussian_dropout_2/StatefulPartitionedCall*gaussian_dropout_2/StatefulPartitionedCall2X
*gaussian_dropout_3/StatefulPartitionedCall*gaussian_dropout_3/StatefulPartitionedCall2P
&gaussian_noise/StatefulPartitionedCall&gaussian_noise/StatefulPartitionedCall2T
(gaussian_noise_1/StatefulPartitionedCall(gaussian_noise_1/StatefulPartitionedCall2T
(gaussian_noise_2/StatefulPartitionedCall(gaussian_noise_2/StatefulPartitionedCall2T
(gaussian_noise_3/StatefulPartitionedCall(gaussian_noise_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
M
1__inference_gaussian_noise_3_layer_call_fn_979467

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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_9787492
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
j
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_978777

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
?M
?
__inference__traced_save_979651
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?
~
)__inference_conv2d_1_layer_call_fn_979302

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
GPU 2J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_9785522
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
?

m
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_978690

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
seed2֡?2$
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

k
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_978579

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
seed2??42$
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
?
?
+__inference_sequential_layer_call_fn_979187

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
GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9789112
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
j
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_979342

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
?
j
1__inference_gaussian_noise_2_layer_call_fn_979392

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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_9786622
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
?
h
/__inference_gaussian_noise_layer_call_fn_979252

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
GPU 2J 8? *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_9784962
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
?@
?
!__inference__wrapped_model_978454
conv2d_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource6
2sequential_conv2d_3_conv2d_readvariableop_resource7
3sequential_conv2d_3_biasadd_readvariableop_resource6
2sequential_dense_mlcmatmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?*sequential/conv2d_3/BiasAdd/ReadVariableOp?)sequential/conv2d_3/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?)sequential/dense/MLCMatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2D	MLCConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R*
num_args *
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	R2
sequential/conv2d/BiasAdd?
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	R2
sequential/conv2d/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:#*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2D	MLCConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	0*
num_args *
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	02
sequential/conv2d_1/BiasAdd?
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????	02
sequential/conv2d_1/Relu?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp?
sequential/conv2d_2/Conv2D	MLCConv2D&sequential/conv2d_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
num_args *
paddingVALID*
strides
2
sequential/conv2d_2/Conv2D?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOp?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
sequential/conv2d_2/BiasAdd?
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
sequential/conv2d_2/Relu?
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02+
)sequential/conv2d_3/Conv2D/ReadVariableOp?
sequential/conv2d_3/Conv2D	MLCConv2D&sequential/conv2d_2/Relu:activations:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
*
num_args *
paddingVALID*
strides
2
sequential/conv2d_3/Conv2D?
*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02,
*sequential/conv2d_3/BiasAdd/ReadVariableOp?
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????.
2
sequential/conv2d_3/BiasAdd?
sequential/conv2d_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????.
2
sequential/conv2d_3/Relu?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape&sequential/conv2d_3/Relu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
)sequential/dense/MLCMatMul/ReadVariableOpReadVariableOp2sequential_dense_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)sequential/dense/MLCMatMul/ReadVariableOp?
sequential/dense/MLCMatMul	MLCMatMul#sequential/flatten/Reshape:output:01sequential/dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MLCMatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd$sequential/dense/MLCMatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense/Sigmoid?
IdentityIdentitysequential/dense/Sigmoid:y:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2X
*sequential/conv2d_3/BiasAdd/ReadVariableOp*sequential/conv2d_3/BiasAdd/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/MLCMatMul/ReadVariableOp)sequential/dense/MLCMatMul/ReadVariableOp:] Y
/
_output_shapes
:?????????T
&
_user_specified_nameconv2d_input
?
D
(__inference_flatten_layer_call_fn_979503

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
GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_9787962
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
?

?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_978552

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
?
j
1__inference_gaussian_dropout_layer_call_fn_979277

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
GPU 2J 8? *U
fPRN
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_9785242
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

k
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_979383

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
seed2Ȧ?2$
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
|
'__inference_conv2d_layer_call_fn_979232

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
GPU 2J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_9784692
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

?
A__inference_dense_layer_call_and_return_conditional_losses_979514

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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_978718

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
?7
?
F__inference_sequential_layer_call_and_return_conditional_losses_978974

inputs
conv2d_978939
conv2d_978941
conv2d_1_978946
conv2d_1_978948
conv2d_2_978953
conv2d_2_978955
conv2d_3_978960
conv2d_3_978962
dense_978968
dense_978970
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_978939conv2d_978941*
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
GPU 2J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_9784692 
conv2d/StatefulPartitionedCall?
gaussian_noise/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_9785002 
gaussian_noise/PartitionedCall?
 gaussian_dropout/PartitionedCallPartitionedCall'gaussian_noise/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_9785282"
 gaussian_dropout/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)gaussian_dropout/PartitionedCall:output:0conv2d_1_978946conv2d_1_978948*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_9785522"
 conv2d_1/StatefulPartitionedCall?
 gaussian_noise_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_9785832"
 gaussian_noise_1/PartitionedCall?
"gaussian_dropout_1/PartitionedCallPartitionedCall)gaussian_noise_1/PartitionedCall:output:0*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_9786112$
"gaussian_dropout_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_1/PartitionedCall:output:0conv2d_2_978953conv2d_2_978955*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_9786352"
 conv2d_2/StatefulPartitionedCall?
 gaussian_noise_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_9786662"
 gaussian_noise_2/PartitionedCall?
"gaussian_dropout_2/PartitionedCallPartitionedCall)gaussian_noise_2/PartitionedCall:output:0*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_9786942$
"gaussian_dropout_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall+gaussian_dropout_2/PartitionedCall:output:0conv2d_3_978960conv2d_3_978962*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_9787182"
 conv2d_3/StatefulPartitionedCall?
 gaussian_noise_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_9787492"
 gaussian_noise_3/PartitionedCall?
"gaussian_dropout_3/PartitionedCallPartitionedCall)gaussian_noise_3/PartitionedCall:output:0*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_9787772$
"gaussian_dropout_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall+gaussian_dropout_3/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_9787962
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_978968dense_978970*
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
GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9788152
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
/
_output_shapes
:?????????T
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_978934
conv2d_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9789112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????T
&
_user_specified_nameconv2d_input
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_978796

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
?

?
B__inference_conv2d_layer_call_and_return_conditional_losses_979223

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
?
f
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_979247

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
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_979498

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
?
{
&__inference_dense_layer_call_fn_979523

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
GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9788152
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
?
l
3__inference_gaussian_dropout_3_layer_call_fn_979487

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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_9787732
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

m
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_978607

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
seed2??2$
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
?
O
3__inference_gaussian_dropout_3_layer_call_fn_979492

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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_9787772
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
M
1__inference_gaussian_dropout_layer_call_fn_979282

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
GPU 2J 8? *U
fPRN
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_9785282
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
?
h
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_979387

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
h
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_979457

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
?
j
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_979412

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
?

k
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_978745

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
?

m
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_979408

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
?G
?
F__inference_sequential_layer_call_and_return_conditional_losses_978832
conv2d_input
conv2d_978480
conv2d_978482
conv2d_1_978563
conv2d_1_978565
conv2d_2_978646
conv2d_2_978648
conv2d_3_978729
conv2d_3_978731
dense_978826
dense_978828
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?(gaussian_dropout/StatefulPartitionedCall?*gaussian_dropout_1/StatefulPartitionedCall?*gaussian_dropout_2/StatefulPartitionedCall?*gaussian_dropout_3/StatefulPartitionedCall?&gaussian_noise/StatefulPartitionedCall?(gaussian_noise_1/StatefulPartitionedCall?(gaussian_noise_2/StatefulPartitionedCall?(gaussian_noise_3/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_978480conv2d_978482*
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
GPU 2J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_9784692 
conv2d/StatefulPartitionedCall?
&gaussian_noise/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_9784962(
&gaussian_noise/StatefulPartitionedCall?
(gaussian_dropout/StatefulPartitionedCallStatefulPartitionedCall/gaussian_noise/StatefulPartitionedCall:output:0'^gaussian_noise/StatefulPartitionedCall*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_9785242*
(gaussian_dropout/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1gaussian_dropout/StatefulPartitionedCall:output:0conv2d_1_978563conv2d_1_978565*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_9785522"
 conv2d_1/StatefulPartitionedCall?
(gaussian_noise_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0)^gaussian_dropout/StatefulPartitionedCall*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_9785792*
(gaussian_noise_1/StatefulPartitionedCall?
*gaussian_dropout_1/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_1/StatefulPartitionedCall:output:0)^gaussian_noise_1/StatefulPartitionedCall*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_9786072,
*gaussian_dropout_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_1/StatefulPartitionedCall:output:0conv2d_2_978646conv2d_2_978648*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_9786352"
 conv2d_2/StatefulPartitionedCall?
(gaussian_noise_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0+^gaussian_dropout_1/StatefulPartitionedCall*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_9786622*
(gaussian_noise_2/StatefulPartitionedCall?
*gaussian_dropout_2/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_2/StatefulPartitionedCall:output:0)^gaussian_noise_2/StatefulPartitionedCall*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_9786902,
*gaussian_dropout_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3gaussian_dropout_2/StatefulPartitionedCall:output:0conv2d_3_978729conv2d_3_978731*
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
GPU 2J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_9787182"
 conv2d_3/StatefulPartitionedCall?
(gaussian_noise_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0+^gaussian_dropout_2/StatefulPartitionedCall*
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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_9787452*
(gaussian_noise_3/StatefulPartitionedCall?
*gaussian_dropout_3/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_3/StatefulPartitionedCall:output:0)^gaussian_noise_3/StatefulPartitionedCall*
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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_9787732,
*gaussian_dropout_3/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall3gaussian_dropout_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_9787962
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_978826dense_978828*
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
GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9788152
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall)^gaussian_dropout/StatefulPartitionedCall+^gaussian_dropout_1/StatefulPartitionedCall+^gaussian_dropout_2/StatefulPartitionedCall+^gaussian_dropout_3/StatefulPartitionedCall'^gaussian_noise/StatefulPartitionedCall)^gaussian_noise_1/StatefulPartitionedCall)^gaussian_noise_2/StatefulPartitionedCall)^gaussian_noise_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(gaussian_dropout/StatefulPartitionedCall(gaussian_dropout/StatefulPartitionedCall2X
*gaussian_dropout_1/StatefulPartitionedCall*gaussian_dropout_1/StatefulPartitionedCall2X
*gaussian_dropout_2/StatefulPartitionedCall*gaussian_dropout_2/StatefulPartitionedCall2X
*gaussian_dropout_3/StatefulPartitionedCall*gaussian_dropout_3/StatefulPartitionedCall2P
&gaussian_noise/StatefulPartitionedCall&gaussian_noise/StatefulPartitionedCall2T
(gaussian_noise_1/StatefulPartitionedCall(gaussian_noise_1/StatefulPartitionedCall2T
(gaussian_noise_2/StatefulPartitionedCall(gaussian_noise_2/StatefulPartitionedCall2T
(gaussian_noise_3/StatefulPartitionedCall(gaussian_noise_3/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????T
&
_user_specified_nameconv2d_input
?
O
3__inference_gaussian_dropout_2_layer_call_fn_979422

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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_9786942
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
?
h
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_978583

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
?
h
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_979272

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
h
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_979317

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

k
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_978524

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
seed2?ݵ2$
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
?
M
1__inference_gaussian_noise_1_layer_call_fn_979327

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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_9785832
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
?
j
1__inference_gaussian_noise_1_layer_call_fn_979322

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
GPU 2J 8? *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_9785792
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
??
?
"__inference__traced_restore_979766
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias&
"assignvariableop_6_conv2d_3_kernel$
 assignvariableop_7_conv2d_3_bias#
assignvariableop_8_dense_kernel!
assignvariableop_9_dense_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate,
(assignvariableop_15_adam_conv2d_kernel_m*
&assignvariableop_16_adam_conv2d_bias_m.
*assignvariableop_17_adam_conv2d_1_kernel_m,
(assignvariableop_18_adam_conv2d_1_bias_m.
*assignvariableop_19_adam_conv2d_2_kernel_m,
(assignvariableop_20_adam_conv2d_2_bias_m.
*assignvariableop_21_adam_conv2d_3_kernel_m,
(assignvariableop_22_adam_conv2d_3_bias_m+
'assignvariableop_23_adam_dense_kernel_m)
%assignvariableop_24_adam_dense_bias_m,
(assignvariableop_25_adam_conv2d_kernel_v*
&assignvariableop_26_adam_conv2d_bias_v.
*assignvariableop_27_adam_conv2d_1_kernel_v,
(assignvariableop_28_adam_conv2d_1_bias_v.
*assignvariableop_29_adam_conv2d_2_kernel_v,
(assignvariableop_30_adam_conv2d_2_bias_v.
*assignvariableop_31_adam_conv2d_3_kernel_v,
(assignvariableop_32_adam_conv2d_3_bias_v+
'assignvariableop_33_adam_dense_kernel_v)
%assignvariableop_34_adam_dense_bias_v
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
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_conv2d_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_conv2d_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv2d_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv2d_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv2d_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_conv2d_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_dense_bias_vIdentity_34:output:0"/device:CPU:0*
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
?
j
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_978694

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
?
?
+__inference_sequential_layer_call_fn_979212

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
GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9789742
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
?
l
3__inference_gaussian_dropout_2_layer_call_fn_979417

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
GPU 2J 8? *W
fRRP
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_9786902
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

i
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_979243

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
?

k
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_979268

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
?	
?
+__inference_sequential_layer_call_fn_978997
conv2d_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9789742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????T::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????T
&
_user_specified_nameconv2d_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
conv2d_input=
serving_default_conv2d_input:0?????????T9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ڌ
?S
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
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?O
_tf_keras_sequential?O{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [1, 35]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_1", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_2", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_3", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 84, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [1, 35]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_1", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_2", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_3", "trainable": true, "dtype": "float32", "stddev": 0.1}}, {"class_name": "GaussianDropout", "config": {"name": "gaussian_dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 84, 1]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
regularization_losses
 trainable_variables
!	variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout", "trainable": true, "dtype": "float32", "rate": 0.1}}
?


#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [1, 35]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 82, 30]}}
?
)regularization_losses
*trainable_variables
+	variables
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_1", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
-regularization_losses
.trainable_variables
/	variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1}}
?


1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [7, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 48, 30]}}
?
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_2", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
;regularization_losses
<trainable_variables
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1}}
?


?kernel
@bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 84, 1]}, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 48, 30]}}
?
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_3", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianDropout", "name": "gaussian_dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5}}
?
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 460}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 460]}}
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
regularization_losses

\layers
trainable_variables
]non_trainable_variables
	variables
^metrics
_layer_regularization_losses
`layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
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
regularization_losses

alayers
trainable_variables
bnon_trainable_variables
	variables
cmetrics
dlayer_regularization_losses
elayer_metrics
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
regularization_losses

flayers
trainable_variables
gnon_trainable_variables
	variables
hmetrics
ilayer_regularization_losses
jlayer_metrics
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
regularization_losses

klayers
 trainable_variables
lnon_trainable_variables
!	variables
mmetrics
nlayer_regularization_losses
olayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'#2conv2d_1/kernel
:2conv2d_1/bias
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
%regularization_losses

players
&trainable_variables
qnon_trainable_variables
'	variables
rmetrics
slayer_regularization_losses
tlayer_metrics
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
)regularization_losses

ulayers
*trainable_variables
vnon_trainable_variables
+	variables
wmetrics
xlayer_regularization_losses
ylayer_metrics
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
-regularization_losses

zlayers
.trainable_variables
{non_trainable_variables
/	variables
|metrics
}layer_regularization_losses
~layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_2/kernel
:2conv2d_2/bias
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
3regularization_losses

layers
4trainable_variables
?non_trainable_variables
5	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
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
7regularization_losses
?layers
8trainable_variables
?non_trainable_variables
9	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
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
;regularization_losses
?layers
<trainable_variables
?non_trainable_variables
=	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
2conv2d_3/kernel
:
2conv2d_3/bias
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
Aregularization_losses
?layers
Btrainable_variables
?non_trainable_variables
C	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
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
Eregularization_losses
?layers
Ftrainable_variables
?non_trainable_variables
G	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
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
Iregularization_losses
?layers
Jtrainable_variables
?non_trainable_variables
K	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
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
Mregularization_losses
?layers
Ntrainable_variables
?non_trainable_variables
O	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?2dense/kernel
:2
dense/bias
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
Sregularization_losses
?layers
Ttrainable_variables
?non_trainable_variables
U	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,#2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
.:,2Adam/conv2d_2/kernel/m
 :2Adam/conv2d_2/bias/m
.:,
2Adam/conv2d_3/kernel/m
 :
2Adam/conv2d_3/bias/m
$:"	?2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,#2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
.:,2Adam/conv2d_2/kernel/v
 :2Adam/conv2d_2/bias/v
.:,
2Adam/conv2d_3/kernel/v
 :
2Adam/conv2d_3/bias/v
$:"	?2Adam/dense/kernel/v
:2Adam/dense/bias/v
?2?
!__inference__wrapped_model_978454?
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
annotations? *3?0
.?+
conv2d_input?????????T
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_979121
F__inference_sequential_layer_call_and_return_conditional_losses_978832
F__inference_sequential_layer_call_and_return_conditional_losses_979162
F__inference_sequential_layer_call_and_return_conditional_losses_978870?
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
?2?
+__inference_sequential_layer_call_fn_978997
+__inference_sequential_layer_call_fn_978934
+__inference_sequential_layer_call_fn_979187
+__inference_sequential_layer_call_fn_979212?
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
B__inference_conv2d_layer_call_and_return_conditional_losses_979223?
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
'__inference_conv2d_layer_call_fn_979232?
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
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_979247
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_979243?
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
/__inference_gaussian_noise_layer_call_fn_979257
/__inference_gaussian_noise_layer_call_fn_979252?
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
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_979272
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_979268?
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
1__inference_gaussian_dropout_layer_call_fn_979282
1__inference_gaussian_dropout_layer_call_fn_979277?
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_979293?
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
)__inference_conv2d_1_layer_call_fn_979302?
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
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_979313
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_979317?
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
1__inference_gaussian_noise_1_layer_call_fn_979327
1__inference_gaussian_noise_1_layer_call_fn_979322?
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
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_979342
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_979338?
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
3__inference_gaussian_dropout_1_layer_call_fn_979347
3__inference_gaussian_dropout_1_layer_call_fn_979352?
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_979363?
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
)__inference_conv2d_2_layer_call_fn_979372?
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
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_979387
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_979383?
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
1__inference_gaussian_noise_2_layer_call_fn_979392
1__inference_gaussian_noise_2_layer_call_fn_979397?
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
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_979408
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_979412?
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
3__inference_gaussian_dropout_2_layer_call_fn_979417
3__inference_gaussian_dropout_2_layer_call_fn_979422?
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_979433?
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
)__inference_conv2d_3_layer_call_fn_979442?
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
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_979457
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_979453?
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
1__inference_gaussian_noise_3_layer_call_fn_979467
1__inference_gaussian_noise_3_layer_call_fn_979462?
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
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_979482
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_979478?
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
3__inference_gaussian_dropout_3_layer_call_fn_979487
3__inference_gaussian_dropout_3_layer_call_fn_979492?
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
C__inference_flatten_layer_call_and_return_conditional_losses_979498?
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
(__inference_flatten_layer_call_fn_979503?
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
A__inference_dense_layer_call_and_return_conditional_losses_979514?
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
&__inference_dense_layer_call_fn_979523?
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
$__inference_signature_wrapper_979024conv2d_input"?
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
!__inference__wrapped_model_978454z
#$12?@QR=?:
3?0
.?+
conv2d_input?????????T
? "-?*
(
dense?
dense??????????
D__inference_conv2d_1_layer_call_and_return_conditional_losses_979293l#$7?4
-?*
(?%
inputs?????????	R
? "-?*
#? 
0?????????	0
? ?
)__inference_conv2d_1_layer_call_fn_979302_#$7?4
-?*
(?%
inputs?????????	R
? " ??????????	0?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_979363l127?4
-?*
(?%
inputs?????????	0
? "-?*
#? 
0?????????0
? ?
)__inference_conv2d_2_layer_call_fn_979372_127?4
-?*
(?%
inputs?????????	0
? " ??????????0?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_979433l?@7?4
-?*
(?%
inputs?????????0
? "-?*
#? 
0?????????.

? ?
)__inference_conv2d_3_layer_call_fn_979442_?@7?4
-?*
(?%
inputs?????????0
? " ??????????.
?
B__inference_conv2d_layer_call_and_return_conditional_losses_979223l7?4
-?*
(?%
inputs?????????T
? "-?*
#? 
0?????????	R
? ?
'__inference_conv2d_layer_call_fn_979232_7?4
-?*
(?%
inputs?????????T
? " ??????????	R?
A__inference_dense_layer_call_and_return_conditional_losses_979514]QR0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_dense_layer_call_fn_979523PQR0?-
&?#
!?
inputs??????????
? "???????????
C__inference_flatten_layer_call_and_return_conditional_losses_979498a7?4
-?*
(?%
inputs?????????.

? "&?#
?
0??????????
? ?
(__inference_flatten_layer_call_fn_979503T7?4
-?*
(?%
inputs?????????.

? "????????????
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_979338l;?8
1?.
(?%
inputs?????????	0
p
? "-?*
#? 
0?????????	0
? ?
N__inference_gaussian_dropout_1_layer_call_and_return_conditional_losses_979342l;?8
1?.
(?%
inputs?????????	0
p 
? "-?*
#? 
0?????????	0
? ?
3__inference_gaussian_dropout_1_layer_call_fn_979347_;?8
1?.
(?%
inputs?????????	0
p
? " ??????????	0?
3__inference_gaussian_dropout_1_layer_call_fn_979352_;?8
1?.
(?%
inputs?????????	0
p 
? " ??????????	0?
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_979408l;?8
1?.
(?%
inputs?????????0
p
? "-?*
#? 
0?????????0
? ?
N__inference_gaussian_dropout_2_layer_call_and_return_conditional_losses_979412l;?8
1?.
(?%
inputs?????????0
p 
? "-?*
#? 
0?????????0
? ?
3__inference_gaussian_dropout_2_layer_call_fn_979417_;?8
1?.
(?%
inputs?????????0
p
? " ??????????0?
3__inference_gaussian_dropout_2_layer_call_fn_979422_;?8
1?.
(?%
inputs?????????0
p 
? " ??????????0?
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_979478l;?8
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
N__inference_gaussian_dropout_3_layer_call_and_return_conditional_losses_979482l;?8
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
3__inference_gaussian_dropout_3_layer_call_fn_979487_;?8
1?.
(?%
inputs?????????.

p
? " ??????????.
?
3__inference_gaussian_dropout_3_layer_call_fn_979492_;?8
1?.
(?%
inputs?????????.

p 
? " ??????????.
?
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_979268l;?8
1?.
(?%
inputs?????????	R
p
? "-?*
#? 
0?????????	R
? ?
L__inference_gaussian_dropout_layer_call_and_return_conditional_losses_979272l;?8
1?.
(?%
inputs?????????	R
p 
? "-?*
#? 
0?????????	R
? ?
1__inference_gaussian_dropout_layer_call_fn_979277_;?8
1?.
(?%
inputs?????????	R
p
? " ??????????	R?
1__inference_gaussian_dropout_layer_call_fn_979282_;?8
1?.
(?%
inputs?????????	R
p 
? " ??????????	R?
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_979313l;?8
1?.
(?%
inputs?????????	0
p
? "-?*
#? 
0?????????	0
? ?
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_979317l;?8
1?.
(?%
inputs?????????	0
p 
? "-?*
#? 
0?????????	0
? ?
1__inference_gaussian_noise_1_layer_call_fn_979322_;?8
1?.
(?%
inputs?????????	0
p
? " ??????????	0?
1__inference_gaussian_noise_1_layer_call_fn_979327_;?8
1?.
(?%
inputs?????????	0
p 
? " ??????????	0?
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_979383l;?8
1?.
(?%
inputs?????????0
p
? "-?*
#? 
0?????????0
? ?
L__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_979387l;?8
1?.
(?%
inputs?????????0
p 
? "-?*
#? 
0?????????0
? ?
1__inference_gaussian_noise_2_layer_call_fn_979392_;?8
1?.
(?%
inputs?????????0
p
? " ??????????0?
1__inference_gaussian_noise_2_layer_call_fn_979397_;?8
1?.
(?%
inputs?????????0
p 
? " ??????????0?
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_979453l;?8
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
L__inference_gaussian_noise_3_layer_call_and_return_conditional_losses_979457l;?8
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
1__inference_gaussian_noise_3_layer_call_fn_979462_;?8
1?.
(?%
inputs?????????.

p
? " ??????????.
?
1__inference_gaussian_noise_3_layer_call_fn_979467_;?8
1?.
(?%
inputs?????????.

p 
? " ??????????.
?
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_979243l;?8
1?.
(?%
inputs?????????	R
p
? "-?*
#? 
0?????????	R
? ?
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_979247l;?8
1?.
(?%
inputs?????????	R
p 
? "-?*
#? 
0?????????	R
? ?
/__inference_gaussian_noise_layer_call_fn_979252_;?8
1?.
(?%
inputs?????????	R
p
? " ??????????	R?
/__inference_gaussian_noise_layer_call_fn_979257_;?8
1?.
(?%
inputs?????????	R
p 
? " ??????????	R?
F__inference_sequential_layer_call_and_return_conditional_losses_978832z
#$12?@QRE?B
;?8
.?+
conv2d_input?????????T
p

 
? "%?"
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_978870z
#$12?@QRE?B
;?8
.?+
conv2d_input?????????T
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_979121t
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
F__inference_sequential_layer_call_and_return_conditional_losses_979162t
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
+__inference_sequential_layer_call_fn_978934m
#$12?@QRE?B
;?8
.?+
conv2d_input?????????T
p

 
? "???????????
+__inference_sequential_layer_call_fn_978997m
#$12?@QRE?B
;?8
.?+
conv2d_input?????????T
p 

 
? "???????????
+__inference_sequential_layer_call_fn_979187g
#$12?@QR??<
5?2
(?%
inputs?????????T
p

 
? "???????????
+__inference_sequential_layer_call_fn_979212g
#$12?@QR??<
5?2
(?%
inputs?????????T
p 

 
? "???????????
$__inference_signature_wrapper_979024?
#$12?@QRM?J
? 
C?@
>
conv2d_input.?+
conv2d_input?????????T"-?*
(
dense?
dense?????????