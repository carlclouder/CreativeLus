#pragma warning(disable:4244)
#pragma warning(disable:4251)
#pragma warning(disable:4267)
#pragma warning(disable:4302)
#pragma warning(disable:4305)
#pragma warning(disable:4311)

#pragma once

#ifndef __CL_CREATIVELUSEX_H__
#define __CL_CREATIVELUSEX_H__

#include "CreativeLus.h"

#include "windows.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <random>
#include <map>
#include "assert.h"

#include <SDKDDKVer.h>
#include <iomanip>

#include "../_cl_string/CLString.h"
#include "../_cl_time/CLTime.h"
#include "../_cl_showTool/CLShowTool.h"
#include "../_cl_arrayTemplate/CLArrayTemplate.h"
#include "../_cl_matrix/CLMatrix.h"
#include "../_cl_binSerial/CLBinSerial.h"

using namespace CreativeLus;

#define UseLinkStruct  1
#define UsePFunc       0
#define UseCppAmp      0
#define UseCheckNanInf 0
#define UseCnnTestOpen 0

#if UseCppAmp > 0
#include <amp.h>
#include <amp_math.h>
using namespace concurrency;
#else
//#include <ppl.h>
#endif

Bool writeSamSetsToFile(const BpnnSamSets& org, PCStr file, Bool binMode = true);
Bool readSamSetsFromFile(BpnnSamSets& tag, PCStr lpFile, Bool binMode = true);

Bool readBpnnStructDefFromFile(BpnnStructDef& mod, PCStr file);
BpnnStructDef& writeBpnnStructDefToFile(BpnnStructDef& mod, PCStr file);


#define Ps_EaTool 1		//Er窗口位置标识
#define Ps_CrTool 2		//Cr窗口位置标识
#define Ps_LsTool 20		//Ls窗口位置标识
#define Ps_McTool 30		//Mc窗口位置标识、
#define Ps_StructTool 40		//Struct窗口位置标识、
#define Ps_StructDMcTool 50		//Struct detail 窗口位置标识、

//传递函数调用宏（NAME传递函数名，ValueX参数名，isDt是否计算导函数值 ）
#define _trFunc( NAME , ValueX , isDt ) tf_##NAME((ValueX),(isDt))
#define _trFunca( NAME , ValueX , a , isDt ) tf_##NAME((ValueX),(a),(isDt))
//传递函数调用宏（NAME传递函数名，ValueX参数名）
#define trFunc( NAME , ValueX ) tf_##NAME((ValueX),0)
#define trFuncG( NAME , ValueX ) tf_##NAME##_amp((ValueX),0)
#define trFunca( NAME , ValueX , a ) tf_##NAME((ValueX),(a),0)
//传递函数调用对应导函数宏（NAME传递函数名，ValueX参数名）
#define trFuncD( NAME , ValueX ) tf_##NAME((ValueX),1)
#define trFuncDa( NAME , ValueX , a ) tf_##NAME((ValueX),(a),1)

#define Abs(x) ((x) < 0 ? (x) * (-1) : (x))

#define BPNN_INDEX     0 //全局编号
#define BPNN_LAYER     1 //层数据下标
#define BPNN_POS       2 //节点下标

#define Minimum_Layers 2 //最少网络层数

#define WbDef_MinWi 2 //最小Wi宽度
#define WbDef_MaxBi 3 //最大Bi宽度

#define BPNU_DIMENSION 3 

#define	FG_WB_NotUpdate  0x1 //不跟新Wi,bi
#define	FG_NN_Dropout    0x2 //节点被dropout
#define	FG_WB_BatchNorm  0x4 //权值组合后展开BatchNorm操作

#if UseLinkStruct > 0
struct neuron {
	//Uint id_index;
	//Uint id_lay;
	//Uint id_pos;	
	Byte wcFuncType;
	Byte transFuncType;
	Uint wb;	
	Uint link;
	Byte bitFlag;
	neuron() { reset(); }
	void reset() { ZeroMemory(this, sizeof(neuron)); }
};
#else
#endif

typedef Float(*PTransFunc)(const Float x, const Bool dt);

struct wbpack {
	Uint index;
	Uint nnIds;//起始节点全局号
	Uint nnIde;//终了节点全局号
	Uint wji_Index;
	Uint wjiSi;
	Byte bFlag;//权值更新标记
	wbpack() { reset(); }
	void reset() { 
		ZeroMemory(this, sizeof(wbpack)); 
		nnIds = UINT_MAX;
	}
};
struct layInfo {
	Uint iLayIndex; //层号0开始
	Uint iLayNnCounts;//本层神经元个数
	Uint iLayNnStartIndex; //起始全局编号
	Uint iLayNnEndIndex; //终点全局编号	
	Bool bIsEndLay;//是否是尾层
	Uint iLayWbCounts;//本层权值个数
	Uint iLayWbStartIndex;//本层权值起始全局编号
	layInfo() { reset(); }
	void reset() { ZeroMemory(this, sizeof(layInfo)); }
};
//网络层信息
class BpnnLayInfo :public vector<layInfo> {
public:
	Uint getLayerMaxNnCountsInNet() const;
	Uint getLayerMinNnCountsInNet() const;	
};
struct BpnnGlobleInfo {
	Uint iLayCounts; //总层数
	Uint iNCounts;//总神经元个数
	Uint iMaxLayNCounts;//拥有最多单元的层的单元数
	Uint iMinLayNCounts;//拥有最小单元的层的单元数
	Uint iMaxLayIndex;
	Uint iMinLayIndex;
	BpnnGlobleInfo() { reset(); }
	void reset() { ZeroMemory(this, sizeof(BpnnGlobleInfo)); }
};

struct ExData {
	Uint index = 0;
	Uint wjiIndex = 0;
	void set(Uint _index = 0, Uint _wjiIndex = 0) {
		index = _index;
		wjiIndex = _wjiIndex;
	}
};
typedef unordered_map<Uint, ExData> NeuronExData;

struct BN{
	Int index = -1;
	Float Eu, Ea2, SqrtEa2;
	Float ri, bt;
	Uint mi;
	struct exBN
	{
		Float u = 0, a2 = 1;
		
		Uint startIndex = 0,units = 0, span = 0;
		Uint ci = 0, nSiTimes = 0,base = 0,si = 0;
		vector<Float> xi;
		vector<Float> xi2;
		vector<Float> dy;

		Float dl_du = 0;
		Float dl_da2 = 0;
		Float dsqrta2 = 0;

		Int& index;
		Float& Eu, &Ea2, & SqrtEa2;
		Float& ri, &bt;
		Uint& mi;
		exBN() = delete;
		exBN(const exBN&) = delete;
		explicit exBN(BN* pbn) :
			index(pbn->index),
			Eu(pbn->Eu), Ea2(pbn->Ea2), SqrtEa2(pbn->SqrtEa2),
			ri(pbn->ri), bt(pbn->bt),
			mi(pbn->mi)
		{};
		void reset(Uint _startIndex, Uint _mi, Uint _units, Uint _span) {			
			startIndex = _startIndex;
			units = max(_units, 1);
			span = max(_span, 1) ;
			si = size_t(max(_mi, 1)) * units * span;
			xi.clear(), xi.resize(si, 0);
			xi2.clear(), xi2.resize(si, 0);
			dy.clear(), dy.resize(si, 0);
			u = 0; a2 = 1;
			ci = 0; base = 0;
			nSiTimes = 0;
		};
		Float putXi(Float _xi, Uint index = 0, Uint l = 0) {
			auto pos = base + (index - startIndex) * span + l;
			return xi[pos] = _xi;
		}
		Float getXi(Uint index = 0, Uint l = 0) {
			auto pos = base + (index - startIndex) * span + l;
			//return xi2[pos] = ri * (xi[pos] - u) * pram1 + bt;
			return ri * (xi2[pos] = (xi[pos] - u) * dsqrta2) + bt;
		}

		void forwardUpdate() {
			Float temp = 0;
			auto pxi = xi.data();			
			for (size_t i = 0; i < si; i++)
				temp += pxi[i];
			u = temp / si;			
			a2 = 0;
			for (size_t i = 0; i < si; i++) {
				auto v = xi[i] - u;
				a2 += v * v;
			}
			a2 /= si;
			Eu = (Eu * nSiTimes + u) / (nSiTimes + 1);
			Ea2 = (Ea2 * nSiTimes + a2) / (nSiTimes + 1);
			SqrtEa2 = sqrt(Ea2 * si / (si - 1) + VtEpslon);
			//SqrtEa2 = sqrt(Ea2 + VtEpslon);
			dsqrta2 = 1.0f / sqrt(a2 + VtEpslon);
		}

		void pushGrad(Float dyi, Uint index, Uint l = 0) {
			dy[base + size_t(index - startIndex) * span + l] = dyi;
		}
		void createBackwardParam() {
			Float pra1 = -0.5 * pow(a2 + VtEpslon, -1.5f);
			dl_da2 = 0;
			for (Uint i = 0; i < si; i++) {
				dl_da2 += dy[i] * (xi[i] - u);
			}
			dl_da2 = dl_da2 * ri * pra1;

			//Float pra2 = -1.0f * sqrt(a2 + VtEpslon);
			Float pra3 = 0;
			for (Uint i = 0; i < si; i++) {
				pra3 += (xi[i] - u);
			}
			pra3 = pra3 * (-2.0f) / si;

			dl_du = 0;
			for (Uint i = 0; i < si; i++) {
				dl_du += dy[i];
			}
			dl_du = dl_du * ri * (-1.0f) * dsqrta2 + dl_da2 * pra3;


		}
		Float gradSendOut(Uint index, Uint l = 0) {
			auto pos = base + (index - startIndex) * span + l;
			return dy[pos] * ri * dsqrta2 + dl_da2 * 2.0 * (xi[pos] - u) / si + dl_du / si;
		}
		void backwardUpdate() {
			Float temp = 0, temp2 = 0;
			for (Uint i = 0; i < si; i++) {
				temp += dy[i] * xi2[i];
				temp2 += dy[i];
			}
			ri += temp;
			bt += temp2;

			ci += 1; if (ci >= mi)ci = 0; base = ci * units * span; ++nSiTimes;
		}
	};
	exBN* ex;
	inline void forwardUpdate() { 
		if(ex)ex->forwardUpdate(); 
	}
	inline void createBackwardParam() {
		if (ex)ex->createBackwardParam();
	}
	inline void backwardUpdate() {
		if (ex)ex->backwardUpdate();
	}
	BN(Uint _mi = 1, Float _ri = 1, Float _bt = 0, Float _Eu = 0,Float _Ea2 = 1) {
		ex = nullptr;
		set(_mi , _ri ,_bt,_Eu ,_Ea2);		
	}
	~BN() { releaseEx(); }
	BN& set(Uint _mi = 1, Float _ri = 1, Float _bt = 0, Float _Eu = 0, Float _Ea2 = 1) {
		setBatch(_mi);
		Eu = _Eu;
		Ea2 = _Ea2;
		SqrtEa2 = sqrt(Ea2 + VtEpslon);
		ri = _ri;
		bt = _bt;
		return *this;
	}
	BN& reset(Uint _startIndex, Uint _mi, Uint units, Uint _span) {
		setBatch(_mi);
		Eu = 0;
		SqrtEa2 = Ea2 = 1;
		ri = 1;
		bt = 0;
		if (ex)
			ex->reset(_startIndex, mi, units, _span);
		return *this;
	}
	Bool isNeedToReset(Uint _units, Uint _span) const {
		if (ex->units != _units || ex->span != _span)
			return true;
		else
			return false;
	}
	inline Bool isExReady() const { return ex != nullptr ? true : false; }
	BN& setBatch(Uint _mi = 1) {
		mi = max(_mi, 1);
		return *this;
	}
	BN& releaseEx() {
		//if (ex)
			delete ex;
		ex = nullptr;
		return *this;
	}
	BN& createEx() {
		if (ex == nullptr)
			ex = new exBN(this);
		return *this;
	}
	Float putXi(Float xi, Bool isTrain = false, Uint index = 0, Uint l = 0) {
		return isTrain ? ex->putXi(xi, index, l) : predict(xi);
	}
	inline Float getXi(Uint index = 0, Uint l = 0) {
		return ex->getXi(index, l);
	}
	Float predict(Float xi) {
		return ri * (xi - Eu) / SqrtEa2 + bt;
	}		
};

#if UseLinkStruct > 0
typedef vector<neuron*> BpnnLayer;//网络节点单层指针
typedef vector<BpnnLayer> BpnnMatrix;//网络节点矩阵
#else
#endif

class CLBpExtend;

enum EWS_PERSID
{
	PERS_QUIT,
	PERS_STANDBY,
	PERS_FORD,
	PERS_FORD_PREDICT,
	PERS_GRAD,
	PERS_MODIFY,

	PERS_TURN_NORM,
	PERS_TURN_BN,
	
	PERS_BN_FORD_XI,
	PERS_BN_FORD_UPDATE,
	PERS_BN_FORD_YI,
	PERS_BN_FORD_NO,
	PERS_BN_GRAD_PUSH,
	PERS_BN_GRAD_CREATE,
	PERS_BN_GRAD_SEND,
	PERS_BN_GRAD_UPDATE,
	PERS_BN_GRAD_NO,
	PERS_BN_FORD_PREDICT,

	PERS_NULL,
};
//工作组类
class WorkSvc :public CLTaskSvc {
public:
	CLBpExtend* const bpnn;
	Uint tsi = 0;
	Uint t0_si = 0;
	Uint nFlag = 0;
	Uint samSi = 0;
	Uint wbSi = 0;
	EWS_PERSID* flag = nullptr;
#if UseLinkStruct > 0
	neuron* pclay = 0;
#else
#endif
	Uint pclaySize = 0;
	LONGLONG cc = 0;
	Uint method = 0;
	Uint clayIndex = 0;	
	const Uint m_core = CLTaskSvc::getCpuCoreCounts();
	Uint getCpuCoreCounts() const { return m_core; };

#define usdcci 0
#define usecsl 1
	WorkSvc() = delete;
	explicit WorkSvc(CLBpExtend* __bpnn) :bpnn(__bpnn) {}
	virtual ~WorkSvc() {close();if (flag)delete[] flag;}	

	virtual DWORD run(PCLTaskSvcTrdParam var);
	//设置当前线程组执行任务，并且是阻塞函数
	void setPermissions(EWS_PERSID type) {
		assert(tsi <= nFlag);
		switch (type)
		{
		case PERS_QUIT:
			for (Uint i = 0; i < tsi; i++)flag[i] = type;
			wait();
			for (Uint i = 0; i < nFlag; i++)flag[i] = PERS_STANDBY;
			tsi = 0;
			return;	//	线程组退出，标记值0
		default:
			if (type > PERS_QUIT && type < PERS_NULL) {
				for (Uint i = 0; i < tsi; i++)flag[i] = type; 
				Sleep(0); 
				break;
			}
			for (Uint i = 0; i < tsi; i++)flag[i] = PERS_STANDBY;
			break;
		}
		for (Uint i = 0; i < tsi; i++)
		{
			Uint all = 0;
			while(flag[i] != PERS_STANDBY) {
				//if (++all >= 1000)
				if (++all >= m_core)
					Sleep(0), all = 0;
			}
		}
		return;
	}
	//创建大于3且小于核心数减1的线程
	Bool start() {
		if (getCpuCoreCounts() < 4)//核心线程小于4则没有必要启动线程组
			return false;
#if usecsl > 0
		setPriority(THREAD_PRIORITY_HIGHEST);
#endif

#define USE_MULTI 0
#if USE_MULTI == 0
		auto nsi = CLTaskSvc::start(getCpuCoreCounts() - 1, TRUE);
		size_t siTag = 3;
#else
		auto nsi = CLTaskSvc::start(USE_MULTI, TRUE);
		size_t siTag = 1;
#endif
		cc = (DWORD)pow(10, 5);
		if (flag) {
			if (nsi > nFlag) {
				delete[] flag;
				flag = new EWS_PERSID[nFlag = nsi];
			}
		}
		else {
			flag = new EWS_PERSID[nFlag = nsi];
		}
		tsi = nsi;
		setPermissions(PERS_STANDBY);
		resume();
		if (nsi >= siTag) {
			return true; 
		}
		else {
			close();
			return false;
		}
	}
	virtual void close() {
		setPermissions(PERS_QUIT);
		CLTaskSvc::close();
	}

	Uint getWorkType(Uint samSi) const;
};

//bp神经网络类(预测内核)
class CLBpKernel {
public:
	string bpnnName;//名称及相关
	BpnnGlobleInfo vm_globleInfo;//总信息
	BpnnLayInfo vm_layInfo;//层信息
	Uint vm_inputDim, vm_outputDim;

	vector<wbpack> vm_wbpack;
	VLF vm_wji_Data;
	VLI vm_link_Data;
	VLF vm_yi0;

	vector<BN> vm_bnData;
	Uint vm_bnMi;

	typedef Float(*PLossFunc)(const Float, const Float, const Bool);
	Uint vm_LossFuncType;
	PLossFunc pLossFunc;

	const Float* inVec;
	/*Float* pxi, * pyi;
	NeuronExData* exData;
	Uint xy_span;*/

	void reset() {
		setName();
		inVec = nullptr;
		vm_inputDim = vm_outputDim = 0;
		vm_bnMi = 0;
		clearContainer();
		setLossFunc();
	}

	BpnnLayInfo& updateLayWbRange();

#if UseLinkStruct > 0
	

	void setWcFunc(const Uint index, const Byte _wcFuncType);
	void setTransFunc(const Uint index, const Byte _transFuncType);

#define _Xi_param_def_default const Uint index,const Float* preLayYiData0, const Uint span, const Uint l,\
	NeuronExData* exData = nullptr, const Uint exdataSpan = 1, const Uint exdatal = 0
#define _Xi_param_def const Uint index, const Float* preLayYiData0, const Uint span, const Uint l,\
	NeuronExData* exData, const Uint exdataSpan, const Uint exdatal 
#define _Xi_param  index, preLayYiData0, span, l, exData,exdataSpan,exdatal
	//Float _Xi(_Xi_param_def_default);

#define _Yi_param_def_default const Uint index, const neuron& nn, Float* pxi0, Float* pyi0, const Float* inVec, const Uint span = 1, const Uint l = 0,\
	NeuronExData* exData = nullptr, const Uint exdataSpan = 1, const Uint exdatal = 0
#define _Yi_param_def const Uint index, const neuron& nn,  Float* pxi0, Float* pyi0, const Float* inVec, const Uint span, const Uint l,\
	NeuronExData* exData , const Uint exdataSpan , const Uint exdatal 
#define _Yi_param  index,nn,pxi0,pyi0,inVec,span,l,exData ,exdataSpan,exdatal
	void _Yi(_Yi_param_def_default);

	void _Xi_bn(_Yi_param_def_default);
	void _Yi_bn(_Yi_param_def_default);

	vector<neuron> vm_neuron;
#define _WcFuncParam_def const Uint index,const neuron& nn,const Float* preLayYiData0,const Uint span,const Uint l,NeuronExData* exData, const Uint exdataSpan, const Uint exdatal
	
	Float _wcFunc(_WcFuncParam_def);

	inline const neuron* lastLayStartNn()const {
		return &vm_neuron[lastLayStartNnIndex()];
	}
	inline neuron* lastLayStartNn() {
		return &vm_neuron[lastLayStartNnIndex()];
	}
	inline const neuron* LayStartNn(Uint layerIndex)const {
		return &vm_neuron[LayStartNnIndex(layerIndex)];
	}
	inline neuron* LayStartNn(Uint layerIndex) {
		return &vm_neuron[LayStartNnIndex(layerIndex)];
	}

	//传递函数
	Float activate_function(const Byte trType, const Float x);
	//自适应检查id值正确性并修正，必须在push_back后调用,调用主体为vt中的对象。找到本节点在整个网络中的id坐标
	static void fitId(neuron& nn, Uint globleIndex, Uint layIndex, Uint cindex);
	
#else
#endif		

	CLBpKernel() { reset(); }

	//函数会用全局的定义覆盖调网络，请不要再BpnnStructDef模式下调用避免结构失效。
	void updateTransFunc(Uint hideTransType,Uint outTransType);
	//做以下事情：1、赋值全局编号；2、构造Wji数据（不构造wji_dt和wji_dt_old）;3、初始化bi和wji
	void createWbByWbDef(wbpack& wb, Uint index, const WbDef& def);
	static void createWbByWbDef(Float* pwji, Float* pbi, const WbDef& def);
	
	//做一次正向计算
	void _predict(Float* pxi,Float * pyi,const Float* inVec);
	CLBpKernel& predict();

	//用内部保存的output和传入的target计算误差，函数必须在调用前运行predict()进行计算才有效
	Float Er(const Float* target, Uint targetSi);
	Float _Er(const Float* pyiData, const Float* target, Uint targetSi);

	Bool writeBpnnToFile(PCStr lpFileFullPathName , Bool binMode = true);
	Bool readBpnnFormFile(PCStr lpFileFullPathName, Bool binMode = true);
	inline Uint neuronCounts() const {
		return vm_globleInfo.iNCounts;
	}
	#define neuronCountsTillLayer(layIndex) \
		((layIndex) < 2 ? 0 : kernel.LayStartNnIndex((layIndex) - 1))
	inline Uint inputDimension() const {
		return vm_inputDim;
	}
	inline Uint outputDimension() const {
		return vm_outputDim;
	}
	inline Uint layerCounts() const {
		return vm_globleInfo.iLayCounts;
	}
	inline Uint hideLayerCounts() const {
		return vm_globleInfo.iLayCounts < 1 ? 0 : vm_globleInfo.iLayCounts - 1;
	}
	inline Uint lastLayIndex() const {
		return vm_globleInfo.iLayCounts - 1;
	}
	inline Uint lastLayStartNnIndex()const {
		return vm_layInfo[lastLayIndex()].iLayNnStartIndex;
	}
	inline Uint LayStartNnIndex(Uint layerIndex)const {
		return vm_layInfo[layerIndex].iLayNnStartIndex;
	}
	inline Uint LayEndNnIndex(Uint layerIndex)const {
		return vm_layInfo[layerIndex].iLayNnEndIndex;
	}
	inline Uint LayNnCounts(Uint layerIndex)const {
		return vm_layInfo[layerIndex].iLayNnCounts;
	}
	
	inline Float* lastLayYi0Start() {
		return &vm_yi0[lastLayStartNnIndex()];
	}
	inline const Float* lastLayYi0Start() const {
		return &vm_yi0[lastLayStartNnIndex()];
	}
	inline Bool isOutlayer(Uint layerIndex) {
		return vm_layInfo[layerIndex].bIsEndLay;
	}
	inline Uint nnLayerIndex(Uint index);
	inline Uint wbLayerIndex(Uint index);

	CLBpKernel& setInput(const Float* inputArray, Uint dataDim);
	CLBpKernel& setInput(const VLF& inputArray);	
	
	//损失函数
	Float loss(const Float y, const Float t);
	Bool getOutput(VLF& out_yi);
	Bool _getOutput(const Float* pyiData, VLF& out_yi);
	void setLossFunc(Byte lossId = LS_MeanSquareLoss);
	void clearContainer(UINT createCounts = 0);
	Float predict(const VLF& _inputVec, VLF* _resultVec = nullptr, VLF* _tagVec = nullptr);
	Float predict(const Float* inputData, Uint inputDimension, VLF* _resultVec = nullptr, VLF* _tagVec = nullptr);

	void makeIndependentDataBuf(VLF& yiData);
	Float predictWithIndependentData(Float* yiData, const VLF& inputVec, VLF* _out_resultVec = nullptr, VLF* tagVec = nullptr);
	
	void setName(PCStr lpName = nullptr);
	PCStr getName() const;
};

//bp神经网络类(训练扩展)
class CLBpExtend{
	friend Bpnn;
public:
	//kernel-----------------------
	CLBpKernel& kernel;
	BpnnLayInfo& vm_layInfo;//层信息
	BpnnGlobleInfo& vm_globleInfo;//总信息
	vector<wbpack>& vm_wbpack;
	VLF& vm_wji_Data;
#if UseLinkStruct > 0
	vector<neuron>& vm_neuron;
#else
#endif
	VLF& vm_yi0;
	VLI& vm_link_Data;
	vector<BN>& vm_bnData;
	vector<Bool> vm_bnLayOpen;

	//own data------------------------
	CLBpExtend* pNetFront, * pNetBack;
	//值变量数据结构	
	Uint hideLayerTrsFunType, outLayerTrsFunType;
	Bool bAutoFit, bSetParam;
	Float g_Er, g_Er_old, g_ls, g_mc, g_ls_old, g_mc_old, g_accuracy, g_DEr, g_DEr_old, A, B;
	Float g_CorrectRate, g_CorrectRate_old;
	Uint hideLayerNumbers, hidePerLayerNumbers, maxTimes, runTimes;
	Uint g_baseLow, g_infiSmalls;
	
	const BpnnStructDef* mode;//自定义隐藏层模式描述对象
	Bool bOpenGraph;
	typedef map<Uint, PVOID> LogoutLine; //显示器对象
	LogoutLine logoutLine;//显示器对象
	VD _er, _ls, _mc, er, ls, mc;//保存外显数据
	VD _cr, cr;//保存外显数据
	WorkSvc work;
	Bool bMutiTrdSupport;
	Bool bAmpSupport;
	BITMAPFILEHEADER m_BtmapFileHdr = { 0 };
	BITMAPINFOHEADER m_BtmapInfoHdr = { 0 };
	BITMAPINFO* pBitmapInfo = 0;
	HGLOBAL hBitmapInfo = 0;
	Uint m_bitmapBufSi = 0;
	Bool bIsCheckLinkBk;
	Bool bIsCheckKeepGoWhenNanInf;
	Uint m_KeepGoWhenNanInfTimes;
	Bool bShowNetAlert;
	Bool bSuspendAlert;
	
	Uint m_netUseToType;
	Float m_CorrectRate;
	Byte m_CorrectRateType;


	const BpnnSamSets* train_samSets;
	Uint train_useSamCounts;
	Bool train_useRandom;//是否启用随机样本对输入
	VLUI train_samUsage;

	const BpnnSamSets* predict_samSets;//保存的预测样本数据集
	Uint predict_useSamCounts;//单词使用个数，0表示全用
	Bool predict_useRandom;//非0状态下是否随机
	VLUI predict_samUsage;
	VLF vm_yi_predict;
	Uint vm_yi_span_predict;

	map<Uint, Float> dpDefineTbl;
	Uint dpRepeatTrainTimes, dpRepeatTrainTimesC;

	map<Uint, Uint> bnDefineTbl;

	Uint gErEquitTimes;//Er连续相等次数
	VLF vm_wji_bk;
	CLBpExtend& reset();

	//globle data-----------------------------------
	//{
	//构造应该有的数据
	
	const BpnnSamSets* vm_samSets;//保存的样本数据集	
	//结构数据------------------------在buildNet中应该完成构造初始化的数据
	

	//训练数据-------------------------在train中应该完成初始化的数据

	VLF vm_drop;
	Uint vm_xy_span;
	VLF vm_xi;
	VLF vm_yi;
	Uint vm_grad_span;
	vector<CLAtomic<Float>> vm_grad;

	NeuronExData vm_neuronExData;

	VLF vm_bi_dt_old_Data;
	VLF vm_wji_dt_Data;
	VLF vm_wji_dt_old_Data;
	
	VLUI* vm_samUsage;

	Float* pxi = 0;
	Float* pyi = 0;
	Uint xy_span = 0;
	NeuronExData* pExdata = 0;
	Uint exdata_span = 0;
	inline void setFordParam(Float* _pxi, Float* _pyi, Uint _xy_span, NeuronExData* _pExdata, Uint _exdata_span) {
		pxi = _pxi;
		pyi = _pyi;
		xy_span = _xy_span;
		pExdata = _pExdata;
		exdata_span = _exdata_span;
	}

	//-----------------------------------
public://未导出函数

	//当createCounts > 0 时候表示清理数据容器后，并建立必要个数的容器,其余容器结构将在buildNet中构造,createCounts = 0会连网络信息头一并清理
	void clearAllDataContainer(Uint createCounts = 0);
	//训练达到目标后执行的释放多余空间的函数
	void releaseTrainDataContainer();
	//xySpan < 0时候保持现状不做修改，xySpan = 0时会清理并释放内存，当xySpan > 0会评价新老值看是否扩充 对数据做无损迁移
	void buildTrainDataContainer(Int xySpan , Int gradSpan);
		
	
	//初始化本节点,设置内部对象默认值
	//void resetNeuron(neuron& nn);
	
#if UseLinkStruct > 0
	//取得权值向量最大最小值，返回权值向量对于0的标准差
	Float getMaxMinWij(neuron& nn, Float& vmin, Float& vmax);
	//传递函数的导函数Y法输入
	Float activate_function_Derv(const Byte trType, const Float y, const Float x);
	//传递函数的导函数
	Float _activate_function_Derv(const Byte trType, const Float x);
	
	void gradient_lay_bn_createParam(const Uint is, const Uint ie);
	void gradient_lay_bn_UpdateParam(const Uint is, const Uint ie);
	void forward_lay_bn_UpdateParam(const Uint is, const Uint ie);

	void forward(const Uint is, const Uint ie);
	void forward_lay(const Uint lay, const Uint ns, const Uint ne, const Uint is, const Uint ie);
	void forward_lay_bn_no(const Uint lay, const Uint ns, const Uint ne, const Uint is, const Uint ie);
	void forward_lay_bn_xi(const Uint lay, const Uint ns, const Uint ne, const Uint is, const Uint ie);
	void forward_lay_bn_yi(const Uint lay, const Uint ns, const Uint ne, const Uint is, const Uint ie);

#define _forward_param_def const Uint index, const Uint lay, const Uint pos,const neuron& nn,const Uint is, const Uint ie
#define _forward_param     index,lay,pos,nn,is,ie
	void _forward_i(_forward_param_def);
	void _forward_i_bn(_forward_param_def);
	void _forward_i_dp(_forward_param_def);
	void _forward_h(_forward_param_def);
	void _forward_h_bn(_forward_param_def);
	void _forward_h_dp(_forward_param_def);

#define _gradient_param_def const Uint index, const Uint lay, const Uint pos,const neuron& nn, const Uint is, const Uint ie
#define _gradient_param     index,lay,pos,nn,is,ie
	//求本层节点的梯度系数	
	void _gradient_o(_gradient_param_def);
	void _gradient_h(_gradient_param_def);
	void _gradient_h_dp(_gradient_param_def);
	void _gradient_i(_gradient_param_def);
	void _gradient_i_dp(_gradient_param_def);

	void _gradient_o_bn(_gradient_param_def);
	void _gradient_h_bn(_gradient_param_def);
	void _gradient_sendOut_bn(_gradient_param_def);

	void gradient(const Uint is, const Uint ie);
	void gradient_lay(const Uint lay, const Uint ns, const Uint ne, const Uint is, const Uint ie);
	void gradient_lay_bn_pushGrad(const Uint lay, const Uint ns, const Uint ne, const Uint is, const Uint ie);
	void gradient_lay_bn_no(const Uint lay, const Uint ns, const Uint ne, const Uint is, const Uint ie);
	void gradient_lay_bn_sendGrad(const Uint lay, const Uint ns, const Uint ne, const Uint is, const Uint ie);

	//清空梯度值为0
	void zeroGradData();

	//内部私有画图函数
	void drawNode(Uint _lay,Uint _pos,Int upNnBase, Int type, HDC hp, Int r, Int lwidth, Int lheight, Int layerNodes, Int nlayers, Uint siFord, Int pr,
		neuron* pNode, Int iStyle, Int iWide, COLORREF cls, HBRUSH hbr,
		Float wmin, Float wmax, Float wqmin, Float wqmax, Bool isDetail);

	//检查Er变化
	void checkErValid();
	//检查多线程是否启动
	void checkMultiThreadStartup();


#else
#endif
	//修正当前节点的权值
	//void modify_wi_and_bi(neuron& nn, Uint nSiSams);
	void modify_wi_and_bi(Uint nSiSams, Uint is, Uint ie);
	void _modify_wi_and_bi(wbpack& wb, Uint nSiSams);
	
	
	void releasBitmapBuf();
	Bool __runByHardwareSpeedup(VLF* pOutEa = nullptr, VLF* pOutLs = nullptr, VLF* pOutMc = nullptr, Bpnn::CbFunStatic _pCbFun = nullptr, PVoid _pIns = nullptr);
	Bool __runByHardwareSpeedup2(VLF* pOutEa = nullptr, VLF* pOutLs = nullptr, VLF* pOutMc = nullptr, Bpnn::CbFunStatic _pCbFun = nullptr, PVoid _pIns = nullptr);

	Bool convergenceHasBeenAchieved();
	//检查网络结构链接是否正常
	Bool checkNeuronLinkSuspended(PVoid _pCbFun = nullptr, PVoid _pIns = nullptr);
	//训练准备输入数据
	Bool prepairTrainSamUsageData();
	//反向Wb结构检查
	void checkWbPackShareLinkRange(PVoid _pCbFun = nullptr,PVoid _pIns = nullptr);
	//检查bn操作层
	void checkWbPackShareBnData(PVoid _pCbFun = nullptr, PVoid _pIns = nullptr);
	
	//损失函数导数
	Float lossDerv(const Float y, const Float t);
	wbpack* newWb();
	//自动调节学习率和动量
	CLBpExtend& autoFitParam();
	//计算当前样本下的累计误差，函数计算由内部loss函数行为决定；
	//bUseLastForwardCalc = true时直接采用当前输出层和目标层的结果计算Er，而不需要再由全部样本带入从新计算；
	Float Er(Bool bUseLastForwardCalc = true);
	//生成权值初值的函数
	Float getDefaultW();
	//内部导出图像函数
	Bool exportGraph(PCStr lpfileName, Int pos);
	
	Bool getBitmapData(HANDLE& hBitmapInfo, BITMAPFILEHEADER& fileHdr, BITMAPINFO*& pdata, Uint& bufSize, Bool bUseDetailMode = false);
	//把样本集和设置到网络首尾端；在多线程模式下，成功返回true，取消返回false；非多线程下什么也不做返回true；
	Bool doPrepair(PVoid _pCbFun, PVoid _pIns);

	//执行一次循环,精度达到返回真true，否则返回false
	Bool _trainOnce();
	
	CLBpExtend& updateDropout();
	CLBpExtend& buildBpnnInfo();//优先以mode来设置



public:			

	CLBpExtend& setDropout(Uint repeatTrainTimes = 0, const DropoutLayerDef& def = DropoutLayerDef());
	CLBpExtend& setBatchNormalization(Uint miniBatch = 0, const BnLayerIndexList& bnLayerList = BnLayerIndexList());

	CLBpExtend& setNetUseTo(Byte type = UT_Classify);

	static Uint getNeuronMenSize();

	CLBpExtend& setMultiThreadSupport(Bool bOpen = false);
	CLBpExtend& setGpuAcceleratedSupport(Bool bOpen = false);

	CLBpExtend& openGraphFlag(Bool bOpen = true);

	CLBpExtend& showGraphParam(Uint maxDataToShow = 0, Int posX = 1, Int posY = 1);

	CLBpExtend& showGraphNetStruct(Bool isShowDetail = false, Int posX = 1, Int posY = 1);

	Bool exportGraphCorrectRate(PCStr lpfileName);
	Bool exportGraphEr(PCStr lpfileName);
	Bool exportGraphLs(PCStr lpfileName);
	Bool exportGraphMc(PCStr lpfileName);

	Uint getSampleCounts() const;
	Uint getRunTimes() const;
	Float getEr() const;
	Float getLs() const;
	Float getMc() const;
	Float getDEr() const;
	Float getAccuracy() const;
	Uint getMaxTimes() const;

	Float getSavedCorrectRate() const;		
	private:
	//默认构造函数
	CLBpExtend(CLBpKernel* pkernel);
	CLBpExtend(const CLBpExtend& bpnn) = delete;
	public:
	~CLBpExtend();

	CLBpExtend& setLayer(Uint hideLayerNumbers , Uint hidePerLayerNumbers );
	
	CLBpExtend& setParam(Float ls , Float er_accuracy , Float mc );

	CLBpExtend& setMaxTimes(Uint iMaxTimes );

	CLBpExtend& setAutoFitLsAndMc(Bool bOpen = false);	

	CLBpExtend& setTransFunc(Byte iBpTypeHide = EBP_TF::TF_Sigmoid, Byte iBpTypeOut = EBP_TF::TF_PRelu);

	CLBpExtend& setWbiDefault(Float W);

	CLBpExtend& setWbiDefault(Float A, Float B);	

	CLBpExtend& setUseRandSample(Bool _isUse = false);

	CLBpExtend& buildNet(Bpnn::CbFunStatic _pCbFun = nullptr, PVoid _pIns = nullptr);

	Bool train(VLF* pOutEa = nullptr, VLF * pOutLs = nullptr, VLF * pOutMc = nullptr, Bpnn::CbFunStatic _pCbFun = nullptr, PVoid _pIns = nullptr);

	//CLBpExtend& logToCmd(Bool bUseDetailedMode = false);	

	CLBpExtend& setSampSets(const BpnnSamSets& tag);

	Float getCorrectRate(const BpnnSamSets* predict,Uint nCounst ,Bool useRandom ,Byte crtype);

	CLBpExtend& setCorrectRateEvaluationModel(Float correctRate = 0, const BpnnSamSets* predict = 0, Uint nCounst = 0, Bool useRandom = false, Byte crtype = CRT_MeanSquareLoss);

	Bool isCorrectRateEvaluationModel() const;

	CLBpExtend& setSampleBatchCounts(Uint nCounts = 1,Bool UseRandomSamp = false);

	CLBpExtend& setStructure(const BpnnStructDef& mod);

	//把内部构件的网络结构以图片方式导出到bitmap文件,bUseDetailMode = true打开绘图细节，表达权值权重等数据
	Bool exportGraphNetStruct(PCStr outFileName,Bool bUseDetailMode = false);
};

#endif