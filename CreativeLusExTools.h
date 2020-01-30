#pragma once

#ifndef __CL_CREATIVELUS_EXTOOLS_H__
#define __CL_CREATIVELUS_EXTOOLS_H__

#include <vector>
#include <map>
#include <string>
#include "assert.h"
#include "fstream"
#include <stdio.h>

using namespace std;

#include "CreativeLus.h"
#ifndef _CL_STDDATAHELPER_DEF_
#define _CL_STDDATAHELPER_DEF_
using namespace cl;
class StdDataHelper {
public:
	static Int reverseInt(Int i)
	{
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = i & 255;
		ch2 = (i >> 8) & 255;
		ch3 = (i >> 16) & 255;
		ch4 = (i >> 24) & 255;
		return((Int)ch1 << 24) + ((Int)ch2 << 16) + ((Int)ch3 << 8) + ch4;
	}
	static void readDataOfMnist(
		string imgfilename,
		string labfilename,
		BpnnSamSets& data_dst,
		Float scMin = -1,
		Float scMax = 1,
		Float labMin = -0.8,
		Float labMax = 0.8,
		Uint padding = 0)
	{
		const Int width_src_image = 28;//ԭʼ��С
		const Int height_src_image = 28;//ԭʼ��С
		const Int x_padding = padding;//���߿��
		const Int y_padding = padding;
		const Uint lab_length = 10;
		const Float scale_min = scMin;
		const Float scale_max = scMax;
		const Float lab_scale_max = labMax;
		const Float lab_scale_min = labMin;

		ifstream file(imgfilename, ios::binary);
		assert(file.is_open());

		Int magic_number = 0;
		Int number_of_images = 0;
		Int n_rows = 0;
		Int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);
		assert(n_rows == height_src_image && n_cols == width_src_image);

		data_dst.clear();
		data_dst.resize(number_of_images, scale_min, lab_scale_min, (n_rows + 2 * y_padding) * (n_cols + 2 * x_padding), lab_length);
		for (Int i = 0; i < number_of_images; ++i) {
			auto pair = data_dst[i];
			for (Int r = 0; r < n_rows; ++r) {
				for (Int c = 0; c < n_cols; ++c) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					pair.iv()[(width_src_image + 2 * x_padding) * (r + y_padding) + c + x_padding] = (temp / 255.0) * (scale_max - scale_min) + scale_min;
				}
			}
		}
		file.close();

		file.open(labfilename, std::ios::binary);
		assert(file.is_open());
		magic_number = 0;
		number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		for (Int i = 0; i < number_of_images; ++i) {
			auto pair = data_dst[i];
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			pair.tv()[temp] = lab_scale_max;
		}
		file.close();
	}
	static void readDataOfCifar10(
		string datafilename,
		BpnnSamSets& data_dst,
		Float scMin = 0,
		Float scMax = 1,
		Float labMin = 0,
		Float labMax = 1,
		Uint padding = 0)
	{
		const Int width_src_image = 32;//ԭʼ��С
		const Int height_src_image = 32;//ԭʼ��С
		const Int deep_src_image = 3;//ԭʼ��С
		const Int x_padding = padding;//���߿��
		const Int y_padding = padding;
		const Uint lab_length = 10;
		const Float scale_min = scMin;
		const Float scale_max = scMax;
		const Float lab_scale_max = labMax;
		const Float lab_scale_min = labMin;

		ifstream file(datafilename, ios::binary);
		assert(file.is_open());

		file.seekg(0, ios::end);
		Int fileSi = file.tellg();
		file.seekg(ios::beg);
		Int number_of_images = fileSi / (3073);

		//data_dst.clear();//������ֻ����
		Uint baseIndex = data_dst.size();
		data_dst.resize(baseIndex + number_of_images, scale_min, lab_scale_min, (height_src_image + 2 * y_padding) * (width_src_image + 2 * x_padding) * deep_src_image, lab_length);
		for (Int i = 0; i < number_of_images; ++i) {
			auto pair = data_dst[baseIndex + i];
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			pair.tv()[temp] = lab_scale_max;
			for (Int d = 0; d < deep_src_image; ++d) {
				Int depSi = (height_src_image + 2 * y_padding) * (width_src_image + 2 * x_padding) * d;
				for (Int r = 0; r < height_src_image; ++r) {
					Int rowSi = (width_src_image + 2 * x_padding) * (r + y_padding);
					for (Int c = 0; c < width_src_image; ++c) {
						temp = 0;
						file.read((char*)&temp, sizeof(temp));
						pair.iv()[depSi + rowSi + c + x_padding] = (temp / 255.0) * (scale_max - scale_min) + scale_min;
					}
				}
			}
		}
		file.close();
	}
};
#endif

#ifndef _CL_CLSPACE_DEF_
#define _CL_CLSPACE_DEF_
template<class Float>
class CLSpaceTemplate {
public:
	struct Point {
		Float x = 0, y = 0, z = 0;
	};
	typedef vector<Point> Polygon;
	struct Line {
		Point pts, pte;
	};
	struct Plane {
		Point pt1, pt2, pt3;
	};
	static Float add(Float v1, Float v2) {
		return v1 + v2;
	}
	static Float sub(Float v1, Float v2) {
		return v1 - v2;
	}
	static Float mul(Float v1, Float v2) {
		return v1 * v2;
	}
	static Float div(Float v1, Float v2) {
		return v1 / v2;
	}
	static bool double_Equal(Float num1, Float num2)
	{
		if ((num1 - num2 > -0.0000001) && (num1 - num2 < 0.0000001))
			return true;
		else
			return false;
	}
	/**
	* �жϵ�ǰ�߶��Ƿ���������ĵ�
	* �������ĵ��Ƿ��ڵ�ǰ����
	*/
	static bool lineIsContainsPoint(const Line& line, const Point& point) {
		bool result = false;
		//�жϸ�����point��˵�1�����߶ε�б���Ƿ�͵�ǰ�߶ε�б����ͬ
		//������point��˵�1�����߶ε�б��k1
		Float k1 = 0;
		bool needjudgment = true;
		if (double_Equal(point.x, line.pts.x)) {
			//k1 = -DBL_MAX;
			needjudgment = false;
		}
		else {
			k1 = div(sub(point.y, line.pts.y), sub(point.x, line.pts.x));
		}
		//��ǰ�߶ε�б��k2
		Float k2 = 0;
		if (double_Equal(line.pte.x, line.pts.x)) {
			//k2 = -DBL_MAX;
			needjudgment = false;
		}
		else {
			k2 = div(sub(line.pte.y, line.pts.y), sub(line.pte.x, line.pts.x));
		}

		if (needjudgment == true) {
			if (double_Equal(k1, k2)) {
				//��б����ͬ�������жϸ�����point��x�Ƿ���pointA.x��pointB.x֮��,���� ��˵���õ��ڵ�ǰ����
				if (sub(point.x, line.pts.x) * sub(point.x, line.pte.x) < 0) {
					result = true;
				}
			}
		}
		return result;
	}
	//���
	static Float mult(const Point& a, const Point& b, const Point& c)
	{
		return (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y);
	}

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

	/**
		* �����߶��Ƿ��뵱ǰ�߶��ཻ
		* �ཻ����true, ���ཻ����false
		*/
	static bool linesIsIntersect(const Line& line1, const Line& line2)
	{
		Point aa = line1.pts;
		Point bb = line1.pte;
		Point cc = line2.pts;
		Point dd = line2.pte;
		if (max(aa.x, bb.x) < min(cc.x, dd.x)) {
			return false;
		}
		if (max(aa.y, bb.y) < min(cc.y, dd.y)) {
			return false;
		}
		if (max(cc.x, dd.x) < min(aa.x, bb.x)) {
			return false;
		}
		if (max(cc.y, dd.y) < min(aa.y, bb.y)) {
			return false;
		}
		if (mult(cc, bb, aa) * mult(bb, dd, aa) < 0) {
			return false;
		}
		if (mult(aa, dd, cc) * mult(dd, bb, cc) < 0) {
			return false;
		}
		return true;
	}
	//����εĸ�����Ҫ���������ӵ������
	static bool pointIsInPolygon(const Point& point, const Polygon& poly) {
		bool result = false;
		int intersectCount = 0;
		// �ж����ݣ����Ӹõ����ҷ�����ˮƽ�����������θ��ߵĽ��㣬��������Ϊ�����������ڲ�
		//����Ҫע�⼸�����������1�����ڱ߻��߶�����;2�����ڱߵ��ӳ�����;3���������ˮƽ�����������ཻ�ڶ�����
		/**
		* ���岽�����£�
		* ѭ�����������߶Σ�
		*  1���жϵ��Ƿ��ڵ�ǰ����(б����ͬ,�Ҹõ��xֵ�������˿ڵ�xֵ֮��),�����򷵻�true
		*  2�������ж��ɸõ㷢����ˮƽ�����Ƿ��뵱ǰ���ཻ,�����ཻ��continue
		*  3�����ཻ,���ж��Ƿ��ཻ�ڶ�����(�ߵĶ˵��Ƿ��ڸ������ˮƽ�Ҳ�).������,����Ϊ�˴��ཻΪ��Խ,������+1 ��continue
		*  4�������ڶ�����,���ж���һ���ߵ�����һ���˵��뵱ǰ�ߵ�����һ���˵��Ƿ�ֲ���ˮƽ���ߵ�����.��������Ϊ�˴��ཻΪ��Խ,������+1.
		*/
		for (size_t i = 0; i < poly.size(); i++) {
			const Point& pointA = poly[i];
			Point pointB;
			Point pointPre;
			//����ǰ�ǵ�һ����,����һ������list��������һ����
			if (i == 0) {
				pointPre = poly[poly.size() - 1];
			}
			else {
				pointPre = poly[i - 1];
			}
			//���Ѿ�ѭ�������һ����,����֮���ӵ��ǵ�һ����
			if (i == (poly.size() - 1)) {
				pointB = poly[0];
			}
			else {
				pointB = poly[i + 1];
			}
			Line line = { pointA, pointB };
			//1���жϵ��Ƿ��ڵ�ǰ����(б����ͬ,�Ҹõ��xֵ�������˿ڵ�xֵ֮��),�����򷵻�true
			bool isAtLine = lineIsContainsPoint(line, point);
			if (isAtLine) {
				return true;
			}
			else {
				//2�������ڱ���,�ж��ɸõ㷢����ˮƽ�����Ƿ��뵱ǰ���ཻ,�����ཻ��continue
				//���ø����ߵ�����һ���˵��xֵ=999,��֤�ߵ�x��Զ������
				Point  radialPoint = { 180, point.y };
				Line radial = { point, radialPoint };
				//�����߶��Ƿ��뵱ǰ�߶��ཻ �ཻ����true
				bool isIntersect = linesIsIntersect(radial, line);
				if (!isIntersect) {
					continue;
				}
				else {
					//3�����ཻ,���ж��Ƿ��ཻ�ڶ�����(�ߵĶ˵��Ƿ��ڸ������ˮƽ�Ҳ�).������,����Ϊ�˴��ཻΪ��Խ,������+1 ��continue
					if (!((pointA.x > point.x) && (double_Equal(pointA.y, point.y))
						|| (pointB.x > point.x) && (double_Equal(pointB.y, point.y)))) {
						intersectCount++;
						continue;
					}
					else {
						//4�������ڶ�����,���ж���һ���ߵ�����һ���˵��뵱ǰ�ߵ�����һ���˵��Ƿ�ֲ���ˮƽ���ߵ�����.��������Ϊ�˴��ཻΪ��Խ,������+1
						if ((pointPre.y - point.y) * (pointB.y - point.y) < 0) {
							intersectCount++;
						}
					}
				}
			}
		}
		result = intersectCount % 2 == 1;
		return result;
	}
};
template class CLSpaceTemplate<float>;
typedef CLSpaceTemplate<float>  CLSpaceF;
template class  CLSpaceTemplate<double>;
typedef CLSpaceTemplate<double> CLSpace;
#endif

#ifndef _CL_BMPHELPER_DEF_
#define _CL_BMPHELPER_DEF_
class CLBmpHelper {
public:
	//��ȡһ��bmp�ļ������������������任�ͷ�ɫ
	static unsigned int readbmp(const char* argv, std::vector<unsigned char>& data)
	{
		FILE* file = 0;
		auto fd = fopen_s(&file, argv, "rb");
		if (0 != fd)
		{
			perror("open bmp file fail");
			throw invalid_argument("open bmp file fail!");
		}
		//����һ��bmp��ͷ
		typedef  unsigned char  U8;
		typedef  unsigned short U16;
		typedef  unsigned int   U32;
#pragma  pack(1)
		struct bmp_header    //����ṹ����Ƕ������Ǹ�ͼ��һ����װ��
		{
			//bmp header
			U8  Signatue[2];   // B  M
			U32 FileSize;     //�ļ���С
			U16 Reserv1;
			U16 Reserv2;
			U32 FileOffset;   //�ļ�ͷƫ����

			//DIB header
			U32 DIBHeaderSize; //DIBͷ��С
			U32 ImageWidth;  //�ļ����
			U32 ImageHight;  //�ļ��߶�
			U16 Planes;
			U16 BPP;  //ÿ�����ص��λ��
			U32 Compression;
			U32 ImageSize;  //ͼ�ļ���С
			U32 XPPM;
			U32 YPPM;
			U32 CCT;
			U32 ICC;
		};
#pragma  pack()
		struct bmp_header  header;
		if (file) {
			fread(&header, sizeof(bmp_header), 1, file);
			if (header.BPP != 8)
				throw invalid_argument("BPP is not 8!");
			data.clear();
			data.resize(header.ImageSize);
			fseek(file, header.FileOffset, 0);
			unsigned char temp = 0;
			for (size_t i = 0; i < header.ImageSize; i++)
			{
				fread(&temp, 1, 1, file);
				//data[(header.ImageSize - i - 1) / header.ImageWidth * header.ImageWidth + i % header.ImageWidth] = (255.0 - (Float)temp) / 255.0 * (max - min) + min;
				data[(header.ImageSize - i - 1) / header.ImageWidth * header.ImageWidth + i % header.ImageWidth] = 255 - temp;
			}
			fclose(file);
			return header.ImageSize;
		}
		else return 0;
	}
};
#endif

#include "windows.h"
#ifndef __CL_TICK_DEF__
#define __CL_TICK_DEF__
//�߾��ȼ�ʱ����
class CLTick {
protected:
	LARGE_INTEGER lis;
	LARGE_INTEGER lie;
	LARGE_INTEGER Freg;
public:
	CLTick() {
		timingStart();
	}
	//��ʼ��ʱ
	CLTick& timingStart() {
		QueryPerformanceFrequency(&Freg);
		QueryPerformanceCounter(&lis);
		return *this;
	}
	//ȡ�ôӼ�ʱ��ʼ����ǰ��ʱ��
	double getSpendTime(bool saveToStart = false) {
		QueryPerformanceCounter(&lie);
		double rt = double(lie.QuadPart - lis.QuadPart) / double(Freg.QuadPart);
		if (saveToStart)lis = lie;
		return rt;
	}
};
#endif

#endif