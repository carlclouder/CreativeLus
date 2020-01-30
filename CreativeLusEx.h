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


#define Ps_EaTool 1		//Er����λ�ñ�ʶ
#define Ps_CrTool 2		//Cr����λ�ñ�ʶ
#define Ps_LsTool 20		//Ls����λ�ñ�ʶ
#define Ps_McTool 30		//Mc����λ�ñ�ʶ��
#define Ps_StructTool 40		//Struct����λ�ñ�ʶ��
#define Ps_StructDMcTool 50		//Struct detail ����λ�ñ�ʶ��

//���ݺ������ú꣨NAME���ݺ�������ValueX��������isDt�Ƿ���㵼����ֵ ��
#define _trFunc( NAME , ValueX , isDt ) tf_##NAME((ValueX),(isDt))
#define _trFunca( NAME , ValueX , a , isDt ) tf_##NAME((ValueX),(a),(isDt))
//���ݺ������ú꣨NAME���ݺ�������ValueX��������
#define trFunc( NAME , ValueX ) tf_##NAME((ValueX),0)
#define trFuncG( NAME , ValueX ) tf_##NAME##_amp((ValueX),0)
#define trFunca( NAME , ValueX , a ) tf_##NAME((ValueX),(a),0)
//���ݺ������ö�Ӧ�������꣨NAME���ݺ�������ValueX��������
#define trFuncD( NAME , ValueX ) tf_##NAME((ValueX),1)
#define trFuncDa( NAME , ValueX , a ) tf_##NAME((ValueX),(a),1)

#define Abs(x) ((x) < 0 ? (x) * (-1) : (x))

#define BPNN_INDEX     0 //ȫ�ֱ��
#define BPNN_LAYER     1 //�������±�
#define BPNN_POS       2 //�ڵ��±�

#define Minimum_Layers 2 //�����������

#define WbDef_MinWi 2 //��СWi���
#define WbDef_MaxBi 3 //���Bi���

#define BPNU_DIMENSION 3 

#define	FG_WB_NotUpdate  0x1 //������Wi,bi
#define	FG_NN_Dropout    0x2 //�ڵ㱻dropout
#define	FG_WB_BatchNorm  0x4 //Ȩֵ��Ϻ�չ��BatchNorm����

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
	Uint nnIds;//��ʼ�ڵ�ȫ�ֺ�
	Uint nnIde;//���˽ڵ�ȫ�ֺ�
	Uint wji_Index;
	Uint wjiSi;
	Byte bFlag;//Ȩֵ���±��
	wbpack() { reset(); }
	void reset() { 
		ZeroMemory(this, sizeof(wbpack)); 
		nnIds = UINT_MAX;
	}
};
struct layInfo {
	Uint iLayIndex; //���0��ʼ
	Uint iLayNnCounts;//������Ԫ����
	Uint iLayNnStartIndex; //��ʼȫ�ֱ��
	Uint iLayNnEndIndex; //�յ�ȫ�ֱ��	
	Bool bIsEndLay;//�Ƿ���β��
	Uint iLayWbCounts;//����Ȩֵ����
	Uint iLayWbStartIndex;//����Ȩֵ��ʼȫ�ֱ��
	layInfo() { reset(); }
	void reset() { ZeroMemory(this, sizeof(layInfo)); }
};
//�������Ϣ
class BpnnLayInfo :public vector<layInfo> {
public:
	Uint getLayerMaxNnCountsInNet() const;
	Uint getLayerMinNnCountsInNet() const;	
};
struct BpnnGlobleInfo {
	Uint iLayCounts; //�ܲ���
	Uint iNCounts;//����Ԫ����
	Uint iMaxLayNCounts;//ӵ����൥Ԫ�Ĳ�ĵ�Ԫ��
	Uint iMinLayNCounts;//ӵ����С��Ԫ�Ĳ�ĵ�Ԫ��
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
typedef vector<neuron*> BpnnLayer;//����ڵ㵥��ָ��
typedef vector<BpnnLayer> BpnnMatrix;//����ڵ����
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
//��������
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
	//���õ�ǰ�߳���ִ�����񣬲�������������
	void setPermissions(EWS_PERSID type) {
		assert(tsi <= nFlag);
		switch (type)
		{
		case PERS_QUIT:
			for (Uint i = 0; i < tsi; i++)flag[i] = type;
			wait();
			for (Uint i = 0; i < nFlag; i++)flag[i] = PERS_STANDBY;
			tsi = 0;
			return;	//	�߳����˳������ֵ0
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
	//��������3��С�ں�������1���߳�
	Bool start() {
		if (getCpuCoreCounts() < 4)//�����߳�С��4��û�б�Ҫ�����߳���
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

//bp��������(Ԥ���ں�)
class CLBpKernel {
public:
	string bpnnName;//���Ƽ����
	BpnnGlobleInfo vm_globleInfo;//����Ϣ
	BpnnLayInfo vm_layInfo;//����Ϣ
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

	//���ݺ���
	Float activate_function(const Byte trType, const Float x);
	//����Ӧ���idֵ��ȷ�Բ�������������push_back�����,��������Ϊvt�еĶ����ҵ����ڵ������������е�id����
	static void fitId(neuron& nn, Uint globleIndex, Uint layIndex, Uint cindex);
	
#else
#endif		

	CLBpKernel() { reset(); }

	//��������ȫ�ֵĶ��帲�ǵ����磬�벻Ҫ��BpnnStructDefģʽ�µ��ñ���ṹʧЧ��
	void updateTransFunc(Uint hideTransType,Uint outTransType);
	//���������飺1����ֵȫ�ֱ�ţ�2������Wji���ݣ�������wji_dt��wji_dt_old��;3����ʼ��bi��wji
	void createWbByWbDef(wbpack& wb, Uint index, const WbDef& def);
	static void createWbByWbDef(Float* pwji, Float* pbi, const WbDef& def);
	
	//��һ���������
	void _predict(Float* pxi,Float * pyi,const Float* inVec);
	CLBpKernel& predict();

	//���ڲ������output�ʹ����target���������������ڵ���ǰ����predict()���м������Ч
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
	
	//��ʧ����
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

//bp��������(ѵ����չ)
class CLBpExtend{
	friend Bpnn;
public:
	//kernel-----------------------
	CLBpKernel& kernel;
	BpnnLayInfo& vm_layInfo;//����Ϣ
	BpnnGlobleInfo& vm_globleInfo;//����Ϣ
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
	//ֵ�������ݽṹ	
	Uint hideLayerTrsFunType, outLayerTrsFunType;
	Bool bAutoFit, bSetParam;
	Float g_Er, g_Er_old, g_ls, g_mc, g_ls_old, g_mc_old, g_accuracy, g_DEr, g_DEr_old, A, B;
	Float g_CorrectRate, g_CorrectRate_old;
	Uint hideLayerNumbers, hidePerLayerNumbers, maxTimes, runTimes;
	Uint g_baseLow, g_infiSmalls;
	
	const BpnnStructDef* mode;//�Զ������ز�ģʽ��������
	Bool bOpenGraph;
	typedef map<Uint, PVOID> LogoutLine; //��ʾ������
	LogoutLine logoutLine;//��ʾ������
	VD _er, _ls, _mc, er, ls, mc;//������������
	VD _cr, cr;//������������
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
	Bool train_useRandom;//�Ƿ������������������
	VLUI train_samUsage;

	const BpnnSamSets* predict_samSets;//�����Ԥ���������ݼ�
	Uint predict_useSamCounts;//����ʹ�ø�����0��ʾȫ��
	Bool predict_useRandom;//��0״̬���Ƿ����
	VLUI predict_samUsage;
	VLF vm_yi_predict;
	Uint vm_yi_span_predict;

	map<Uint, Float> dpDefineTbl;
	Uint dpRepeatTrainTimes, dpRepeatTrainTimesC;

	map<Uint, Uint> bnDefineTbl;

	Uint gErEquitTimes;//Er������ȴ���
	VLF vm_wji_bk;
	CLBpExtend& reset();

	//globle data-----------------------------------
	//{
	//����Ӧ���е�����
	
	const BpnnSamSets* vm_samSets;//������������ݼ�	
	//�ṹ����------------------------��buildNet��Ӧ����ɹ����ʼ��������
	

	//ѵ������-------------------------��train��Ӧ����ɳ�ʼ��������

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
public://δ��������

	//��createCounts > 0 ʱ���ʾ�������������󣬲�������Ҫ����������,���������ṹ����buildNet�й���,createCounts = 0����������Ϣͷһ������
	void clearAllDataContainer(Uint createCounts = 0);
	//ѵ���ﵽĿ���ִ�е��ͷŶ���ռ�ĺ���
	void releaseTrainDataContainer();
	//xySpan < 0ʱ�򱣳���״�����޸ģ�xySpan = 0ʱ�������ͷ��ڴ棬��xySpan > 0����������ֵ���Ƿ����� ������������Ǩ��
	void buildTrainDataContainer(Int xySpan , Int gradSpan);
		
	
	//��ʼ�����ڵ�,�����ڲ�����Ĭ��ֵ
	//void resetNeuron(neuron& nn);
	
#if UseLinkStruct > 0
	//ȡ��Ȩֵ���������Сֵ������Ȩֵ��������0�ı�׼��
	Float getMaxMinWij(neuron& nn, Float& vmin, Float& vmax);
	//���ݺ����ĵ�����Y������
	Float activate_function_Derv(const Byte trType, const Float y, const Float x);
	//���ݺ����ĵ�����
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
	//�󱾲�ڵ���ݶ�ϵ��	
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

	//����ݶ�ֵΪ0
	void zeroGradData();

	//�ڲ�˽�л�ͼ����
	void drawNode(Uint _lay,Uint _pos,Int upNnBase, Int type, HDC hp, Int r, Int lwidth, Int lheight, Int layerNodes, Int nlayers, Uint siFord, Int pr,
		neuron* pNode, Int iStyle, Int iWide, COLORREF cls, HBRUSH hbr,
		Float wmin, Float wmax, Float wqmin, Float wqmax, Bool isDetail);

	//���Er�仯
	void checkErValid();
	//�����߳��Ƿ�����
	void checkMultiThreadStartup();


#else
#endif
	//������ǰ�ڵ��Ȩֵ
	//void modify_wi_and_bi(neuron& nn, Uint nSiSams);
	void modify_wi_and_bi(Uint nSiSams, Uint is, Uint ie);
	void _modify_wi_and_bi(wbpack& wb, Uint nSiSams);
	
	
	void releasBitmapBuf();
	Bool __runByHardwareSpeedup(VLF* pOutEa = nullptr, VLF* pOutLs = nullptr, VLF* pOutMc = nullptr, Bpnn::CbFunStatic _pCbFun = nullptr, PVoid _pIns = nullptr);
	Bool __runByHardwareSpeedup2(VLF* pOutEa = nullptr, VLF* pOutLs = nullptr, VLF* pOutMc = nullptr, Bpnn::CbFunStatic _pCbFun = nullptr, PVoid _pIns = nullptr);

	Bool convergenceHasBeenAchieved();
	//�������ṹ�����Ƿ�����
	Bool checkNeuronLinkSuspended(PVoid _pCbFun = nullptr, PVoid _pIns = nullptr);
	//ѵ��׼����������
	Bool prepairTrainSamUsageData();
	//����Wb�ṹ���
	void checkWbPackShareLinkRange(PVoid _pCbFun = nullptr,PVoid _pIns = nullptr);
	//���bn������
	void checkWbPackShareBnData(PVoid _pCbFun = nullptr, PVoid _pIns = nullptr);
	
	//��ʧ��������
	Float lossDerv(const Float y, const Float t);
	wbpack* newWb();
	//�Զ�����ѧϰ�ʺͶ���
	CLBpExtend& autoFitParam();
	//���㵱ǰ�����µ��ۼ��������������ڲ�loss������Ϊ������
	//bUseLastForwardCalc = trueʱֱ�Ӳ��õ�ǰ������Ŀ���Ľ������Er��������Ҫ����ȫ������������¼��㣻
	Float Er(Bool bUseLastForwardCalc = true);
	//����Ȩֵ��ֵ�ĺ���
	Float getDefaultW();
	//�ڲ�����ͼ����
	Bool exportGraph(PCStr lpfileName, Int pos);
	
	Bool getBitmapData(HANDLE& hBitmapInfo, BITMAPFILEHEADER& fileHdr, BITMAPINFO*& pdata, Uint& bufSize, Bool bUseDetailMode = false);
	//�������������õ�������β�ˣ��ڶ��߳�ģʽ�£��ɹ�����true��ȡ������false���Ƕ��߳���ʲôҲ��������true��
	Bool doPrepair(PVoid _pCbFun, PVoid _pIns);

	//ִ��һ��ѭ��,���ȴﵽ������true�����򷵻�false
	Bool _trainOnce();
	
	CLBpExtend& updateDropout();
	CLBpExtend& buildBpnnInfo();//������mode������



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
	//Ĭ�Ϲ��캯��
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

	//���ڲ�����������ṹ��ͼƬ��ʽ������bitmap�ļ�,bUseDetailMode = true�򿪻�ͼϸ�ڣ����ȨֵȨ�ص�����
	Bool exportGraphNetStruct(PCStr outFileName,Bool bUseDetailMode = false);
};

#endif