"?G
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE133333??@A33333??@a܃B?CI??i܃B?CI???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(133333??@933333??@A33333??@I33333??@a???-???i?1??h???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1?????)u@9?????)u@A?????)u@I?????)u@a'`??̀??i?????????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     Hq@9     Hq@A     Hq@I     Hq@a?H'8??i@?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1fffff?e@9fffff?e@Afffff?e@Ifffff?e@aPZ?y?	??i????????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(133333Sc@933333Sc@A33333Sc@I33333Sc@a?q?Z????i ?H.????Unknown
^HostGatherV2"GatherV2(1fffff?D@9fffff?D@Afffff?D@Ifffff?D@aa??u??i??N#???Unknown
}	HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1????̌D@9????̌D@A????̌D@I????̌D@a?e?i????i?"?Y-k???Unknown
o
HostSoftmax"sequential/dense_1/Softmax(1??????C@9??????C@A??????C@I??????C@a???O??i???l????Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?;@9     ?;@A3333336@I3333336@a}[f+is?ii?e?>????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1     ?4@9     ?4@A     ?4@I     ?4@a?V
d??q?i?-?????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(13333335@93333335@A?????2@I?????2@a??.???o?i?e????Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1??????.@9??????.@A??????.@I??????.@ab????j?i???[?5???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1333333,@9333333,@A333333,@I333333,@a"{? ?h?i??\TN???Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1ffffff+@9ffffff+@Affffff+@Iffffff+@a??4???g?i?'WMIf???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??????*@9??????*@A??????*@I??????*@a???F?ng?ix????}???Unknown
iHostWriteSummary"WriteSummary(1??????)@9??????)@A??????)@I??????)@a?d?bf?i?X?????Unknown?
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1ffffff&@9ffffff&@Affffff&@Iffffff&@aG??Bʕc?iu??ǯ????Unknown
[HostAddV2"Adam/add(1ffffff"@9ffffff"@Affffff"@Iffffff"@a???mx`?iN?@Ʒ???Unknown
cHostDataset"Iterator::Root(1     ?8@9     ?8@A333333"@I333333"@as7??h?_?i?_??????Unknown
gHostStridedSlice"strided_slice(1??????!@9??????!@A??????!@I??????!@aMD?NX _?iy? @????Unknown
YHostPow"Adam/Pow(1ffffff!@9ffffff!@Affffff!@Iffffff!@a$Q?Gm^?i??~?v????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1??????@9??????@A??????@I??????@ab????Z?i??????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @aY>??<{X?i???]+ ???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      @9      @A      @I      @aw?铻V?i<?'????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1??????@9??????@A??????@I??????@a?d?bV?i>K?-????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a(?d-sUU?i??^?d!???Unknown
ZHostArgMax"ArgMax(1??????@9??????@A??????@I??????@a??b?T?i????+???Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a??qR?S?iF????5???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a??qR?S?iϊ8k?????Unknown
w HostDataset""Iterator::Root::ParallelMapV2::Zip(1?????L@9?????L@A??????@I??????@a ?	???R?i?+?I???Unknown
?!HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a?+!?1?R?i8 a[R???Unknown
?"HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a??¿N?in??P?Y???Unknown
?#HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@a??¿N?i?,?@ea???Unknown
e$Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@aC?v??L?i`ʨ??h???Unknown?
V%HostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@aC?v??L?ihj?o???Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a??m?J?i"??+av???Unknown
v'HostCast"$sparse_categorical_crossentropy/Cast(1??????@9??????@A??????@I??????@a?$3?]?I?i?uI??|???Unknown
?(HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1??????@9??????@A??????@I??????@a?1b1M.I?iwΕ%????Unknown
?)HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1333333@9333333@A333333@I333333@a1K?u,?G?i?>?!????Unknown
~*HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@a
X?G?i`:?h܎???Unknown
?+HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1ffffff
@9ffffff
@Affffff
@Iffffff
@a
X?G?i66???????Unknown
?,HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1??????	@9??????	@A??????	@I??????	@a?d?bF?iϽ?2:????Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1??????@9??????@A??????@I??????@a?qM\??E?i+ф??????Unknown
X.HostCast"Cast_4(1??????@9??????@A??????@I??????@a?qM\??E?i??[?????Unknown
?/HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9????????A??????@I????????a?qM\??E?i??2o}????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_1(1      @9      @A      @I      @a?~|???D?i??i?????Unknown
?1HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1??????@9??????@A??????@I??????@a ?	???B?il?ku????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_4(1??????@9??????@A??????@I??????@a??8??/B?i???????Unknown
]3HostCast"Adam/Cast_1(1      @9      @A      @I      @aҾg)?|A?i?)`????Unknown
[4HostPow"
Adam/Pow_1(1      @9      @A      @I      @aҾg)?|A?ix[bO?????Unknown
t5HostReadVariableOp"Adam/Cast/ReadVariableOp(1333333@9333333@A333333@I333333@a?˖ˈ?@?i+A???????Unknown
V6HostCast"Cast(1ffffff@9ffffff@Affffff@Iffffff@a???mx@?i???O?????Unknown
v7HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1??????@9??????@A??????@I??????@a?????>?iگ?)?????Unknown
?8HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1??????=@9??????=@A?????? @I?????? @ak?Gd?`=?i?8??|????Unknown
`9HostDivNoNan"
div_no_nan(1?????? @9?????? @A?????? @I?????? @ak?Gd?`=?i???U(????Unknown
X:HostEqual"Equal(1ffffff??9ffffff??Affffff??Iffffff??a??m?:?iWbK?z????Unknown
?;HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1ffffff??9ffffff??Affffff??Iffffff??a??m?:?i?	q?????Unknown
w<HostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333??9333333??A333333??I333333??a1K?u,?7?i㺗v?????Unknown
w=HostReadVariableOp"div_no_nan_1/ReadVariableOp(1????????9????????A????????I????????a?d?b6?i????????Unknown
?>HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a?d?b6?i}B??^????Unknown
X?HostCast"Cast_2(1      ??9      ??A      ??I      ??a?~|???4?i?v?????Unknown
X@HostCast"Cast_3(1      ??9      ??A      ??I      ??a?~|???4?i??E??????Unknown
uAHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?~|???4?i-??q=????Unknown
?BHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a?~|???4?i????????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff??9ffffff??Affffff??Iffffff??aG??Bʕ3?i?M?O????Unknown
?DHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a??8??/2?i&?~??????Unknown
vEHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1333333??9333333??A333333??I333333??a?˖ˈ?0?i?5?ή????Unknown
oFHostReadVariableOp"Adam/ReadVariableOp(1333333??9333333??A333333??I333333??a?˖ˈ?0?iب???????Unknown
bGHostDivNoNan"div_no_nan_1(1333333??9333333??A333333??I333333??a?˖ˈ?0?i??0?????Unknown
THHostMul"Mul(1      ??9      ??A      ??I      ??a?????+?i??٠????Unknown
?IHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?????+?iq0??`????Unknown
yJHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1????????9????????A????????I????????a?d?b&?iW?[??????Unknown
aKHostIdentity"Identity(1ffffff??9ffffff??Affffff??Iffffff??aG??Bʕ#?i      ???Unknown?*?F
uHostFlushSummaryWriter"FlushSummaryWriter(133333??@933333??@A33333??@I33333??@a???`x??i???`x???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1?????)u@9?????)u@A?????)u@I?????)u@a??q?q??itَ??????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     Hq@9     Hq@A     Hq@I     Hq@a)??Q???i?b?vt???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1fffff?e@9fffff?e@Afffff?e@Ifffff?e@a^??^????i?O2?h????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(133333Sc@933333Sc@A33333Sc@I33333Sc@a??Bv!???i????L????Unknown
^HostGatherV2"GatherV2(1fffff?D@9fffff?D@Afffff?D@Ifffff?D@a?e??;???i??o??f???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1????̌D@9????̌D@A????̌D@I????̌D@a$b?V???i???H1????Unknown
oHostSoftmax"sequential/dense_1/Softmax(1??????C@9??????C@A??????C@I??????C@a9?xJ!???i??S?z???Unknown
?	HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?;@9     ?;@A3333336@I3333336@aޖ?????i?ع?????Unknown
?
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1     ?4@9     ?4@A     ?4@I     ?4@aC?Ahd???i+?yK????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(13333335@93333335@A?????2@I?????2@aֺF??~?i?????J???Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1??????.@9??????.@A??????.@I??????.@a????Wz?il??Y???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1333333,@9333333,@A333333,@I333333,@a?<??Fx?i?R??????Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1ffffff+@9ffffff+@Affffff+@Iffffff+@a0؍ow?iE;?s????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??????*@9??????*@A??????*@I??????*@a??Pt??v?i??#YK???Unknown
iHostWriteSummary"WriteSummary(1??????)@9??????)@A??????)@I??????)@a??AA?u?i?'?a8???Unknown?
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1ffffff&@9ffffff&@Affffff&@Iffffff&@a9a?c(s?i{Z)f^???Unknown
[HostAddV2"Adam/add(1ffffff"@9ffffff"@Affffff"@Iffffff"@aJD?6yo?i?(6_?}???Unknown
cHostDataset"Iterator::Root(1     ?8@9     ?8@A333333"@I333333"@a?=??!o?i??B????Unknown
gHostStridedSlice"strided_slice(1??????!@9??????!@A??????!@I??????!@a_1??yrn?i.?-{s????Unknown
YHostPow"Adam/Pow(1ffffff!@9ffffff!@Affffff!@Iffffff!@a?$??Q?m?iSf??6????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1??????@9??????@A??????@I??????@a????Wj?i9?V?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a??_?|?g?i?H?Ҁ???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      @9      @A      @I      @a??FR?<f?i??k?!???Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1??????@9??????@A??????@I??????@a??AA?e?i!?Oo?7???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a2H?d?i?^??L???Unknown
ZHostArgMax"ArgMax(1??????@9??????@A??????@I??????@a%t(?/d?i,Jׯ`???Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a?g??c?i~J?/t???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a?g??c?i?h?Ư????Unknown
wHostDataset""Iterator::Root::ParallelMapV2::Zip(1?????L@9?????L@A??????@I??????@a?T?;yb?i;xu)????Unknown
?HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@aMN
??!b?i????J????Unknown
? HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@ar???k]?i?Z׈ ????Unknown
?!HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@ar???k]?i?2?g?????Unknown
e"Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a7?sm\?i? m?????Unknown?
V#HostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@a7?sm\?i??&??????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a_?}??Y?i?????????Unknown
v%HostCast"$sparse_categorical_crossentropy/Cast(1??????@9??????@A??????@I??????@a??s??PY?i?G#6l????Unknown
?&HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1??????@9??????@A??????@I??????@a%?iɤ?X?ii??????Unknown
?'HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1333333@9333333@A333333@I333333@a??U?TCW?i??ʲ^???Unknown
~(HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@aL?Kc,?V?iM?Ȩ"???Unknown
?)HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1ffffff
@9ffffff
@Affffff
@Iffffff
@aL?Kc,?V?i??-??-???Unknown
?*HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1??????	@9??????	@A??????	@I??????	@a??AA?U?i??Na?8???Unknown
t+HostAssignAddVariableOp"AssignAddVariableOp(1??????@9??????@A??????@I??????@a?7?5U?im/^O?C???Unknown
X,HostCast"Cast_4(1??????@9??????@A??????@I??????@a?7?5U?i1?m=N???Unknown
?-HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9????????A??????@I????????a?7?5U?i?f}+?X???Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_1(1      @9      @A      @I      @atz-???T?i??{??b???Unknown
?/HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1??????@9??????@A??????@I??????@a?T?;yR?i\?G#6l???Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1??????@9??????@A??????@I??????@a?Gu?Q?i -u???Unknown
]1HostCast"Adam/Cast_1(1      @9      @A      @I      @aa;?R?Q?i?????}???Unknown
[2HostPow"
Adam/Pow_1(1      @9      @A      @I      @aa;?R?Q?i<U6????Unknown
t3HostReadVariableOp"Adam/Cast/ReadVariableOp(1333333@9333333@A333333@I333333@a?.?0?kP?i?{??k????Unknown
V4HostCast"Cast(1ffffff@9ffffff@Affffff@Iffffff@aJD?6yO?id?tGJ????Unknown
v5HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1??????@9??????@A??????@I??????@a+???N?i?]? ѝ???Unknown
?6HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1??????=@9??????=@A?????? @I?????? @a?????L?is?P& ????Unknown
`7HostDivNoNan"
div_no_nan(1?????? @9?????? @A?????? @I?????? @a?????L?i?0?K/????Unknown
X8HostEqual"Equal(1ffffff??9ffffff??Affffff??Iffffff??a_?}??I?io??H?????Unknown
?9HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1ffffff??9ffffff??Affffff??Iffffff??a_?}??I?i??<F/????Unknown
w:HostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333??9333333??A333333??I333333??a??U?TCG?iRE^ ????Unknown
w;HostReadVariableOp"div_no_nan_1/ReadVariableOp(1????????9????????A????????I????????a??AA?E?i??n\y????Unknown
?<HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a??AA?E?i?~??????Unknown
X=HostCast"Cast_2(1      ??9      ??A      ??I      ??atz-???D?i{1~J????Unknown
X>HostCast"Cast_3(1      ??9      ??A      ??I      ??atz-???D?i?|}?5????Unknown
u?HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??atz-???D?i9?|?W????Unknown
?@HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??atz-???D?i?|Qy????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff??9ffffff??Affffff??Iffffff??a9a?c(C?i?YjjC????Unknown
?BHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a?Gu?A?iB?G??????Unknown
vCHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1333333??9333333??A333333??I333333??a?.?0?k@?i????????Unknown
oDHostReadVariableOp"Adam/ReadVariableOp(1333333??9333333??A333333??I333333??a?.?0?k@?i????????Unknown
bEHostDivNoNan"div_no_nan_1(1333333??9333333??A333333??I333333??a?.?0?k@?i&P??????Unknown
TFHostMul"Mul(1      ??9      ??A      ??I      ??a???QE^;?ie?V?r????Unknown
?GHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a???QE^;?i?? S?????Unknown
yHHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1????????9????????A????????I????????a??AA?5?i?܈??????Unknown
aIHostIdentity"Identity(1ffffff??9ffffff??Affffff??Iffffff??a9a?c(3?i     ???Unknown?2Nvidia GPU (Turing)