// [INFO] Iniciando execução do programa...
// [INFO] Lendo arquivo de entrada...
// [INFO] Tamanho da tabela lida: 104488
// [INFO] Treinando a árvore de decisão...
// Tempo de treinamento: 7928 ms
// [INFO] Testando o modelo...
// TESTANDO O MODELO:

// RESULTADOS:
//   Acertos: 20769 / 20898
//   Acurácia Geral: 99.382716%

// ACURÁCIA POR CLASSE:
//   Cancelled: 96.840796%
//   Diverted: 100.000000%
//   cancelado: 99.985297%

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <map>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t cuda_status = call; \
        if (cuda_status != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(cuda_status) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

// Variável global para tamanho do bloco CUDA
int BLOCK_SIZE = 256;

class Table {
	public:
		vector<string> attrName;
		vector<vector<string> > data;
		vector<vector<string> > attrValueList;
		
		void extractAttrValue() {
			attrValueList.resize(attrName.size());
			for(int j=0; j<attrName.size(); j++) {
				map<string, int> value;
				for(int i=0; i<data.size(); i++) {
					if(j < data[i].size()) {
						value[data[i][j]]=1;
					}
				}
				for(auto iter=value.begin(); iter != value.end(); iter++) {
					attrValueList[j].push_back(iter->first);
				}
			}
		}
};

class Node {
	public:
		int criteriaAttrIndex;
		string attrValue;
		int treeIndex;
		bool isLeaf;
		string label;
		vector<int > children;
		
		Node() {
			isLeaf = false;
		}
};

// Kernel CUDA simplificado e corrigido
__global__ void calculateAttrStats(int* d_attrData, int* d_classData, 
                                   int numSamples, int numClasses, int numAttrValues,
                                   int* d_attrValueCounts,
                                   int* d_attrValueClassCounts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numSamples) {
        int attrVal = d_attrData[idx];
        int classVal = d_classData[idx];
        
        if (attrVal >= 0 && attrVal < numAttrValues && 
            classVal >= 0 && classVal < numClasses) {
            atomicAdd(&d_attrValueCounts[attrVal], 1);
            atomicAdd(&d_attrValueClassCounts[attrVal * numClasses + classVal], 1);
        }
    }
}

class DecisionTree {
	public:
		Table initialTable;
		vector<Node> tree;

		DecisionTree(Table table) {
			initialTable = table;
			initialTable.extractAttrValue();

			Node root;
			root.treeIndex=0;
			tree.push_back(root);
			run(initialTable, 0);
			printTree(0, "");
		}

		string guess(vector<string> row) {
			string label = "";
			int leafNode = dfs(row, 0);
			if(leafNode == -1) {
				return "Unknown";
			}
			label = tree[leafNode].label;
			return label;
		}

		int dfs(vector<string>& row, int here) {
			if(tree[here].isLeaf) {
				return here;
			}

			int criteriaAttrIndex = tree[here].criteriaAttrIndex;
			
			if(criteriaAttrIndex >= row.size()) {
				return -1;
			}

			for(int i=0;i<tree[here].children.size(); i++) {
				int next = tree[here].children[i];
				
				if(next >= tree.size()) {
					continue;
				}

				if (row[criteriaAttrIndex] == tree[next].attrValue) {
					return dfs(row, next);
				}
			}
			return -1;
		}

		void run(Table table, int nodeIndex) {
			if(table.data.size() == 0 || isLeafNode(table) == true) {
				tree[nodeIndex].isLeaf = true;
				if(table.data.size() > 0 && table.data[0].size() > 0) {
					tree[nodeIndex].label = table.data.back().back();
				} else {
					tree[nodeIndex].label = "Unknown";
				}
				return;
			}

			int selectedAttrIndex = getSelectedAttribute(table);
			
			if(selectedAttrIndex == -1) {
				tree[nodeIndex].isLeaf = true;
				tree[nodeIndex].label = getMajorityLabel(table).first;
				return;
			}

			map<string, vector<int> > attrValueMap;
			for(int i=0;i<table.data.size();i++) {
				if(selectedAttrIndex < table.data[i].size()) {
					attrValueMap[table.data[i][selectedAttrIndex]].push_back(i);
				}
			}

			tree[nodeIndex].criteriaAttrIndex = selectedAttrIndex;

			pair<string, int> majority = getMajorityLabel(table);
			if((double)majority.second/table.data.size() > 0.8) {
				tree[nodeIndex].isLeaf = true;
				tree[nodeIndex].label = majority.first;
				return;
			}

			if(selectedAttrIndex >= initialTable.attrValueList.size()) {
				tree[nodeIndex].isLeaf = true;
				tree[nodeIndex].label = majority.first;
				return;
			}

			for(int i=0;i< initialTable.attrValueList[selectedAttrIndex].size(); i++) {
				string attrValue = initialTable.attrValueList[selectedAttrIndex][i];

				Table nextTable;
				vector<int> candi = attrValueMap[attrValue];
				for(int i=0;i<candi.size(); i++) {
					if(candi[i] < table.data.size()) {
						nextTable.data.push_back(table.data[candi[i]]);
					}
				}

				Node nextNode;
				nextNode.attrValue = attrValue;
				nextNode.treeIndex = (int)tree.size();
				tree[nodeIndex].children.push_back(nextNode.treeIndex);
				tree.push_back(nextNode);

				if(nextTable.data.size()==0) {
					nextNode.isLeaf = true;
					nextNode.label = getMajorityLabel(table).first;
					tree[nextNode.treeIndex] = nextNode;
				} else {
					nextTable.attrName = table.attrName;
					run(nextTable, nextNode.treeIndex);
				}
			}
		}

		pair<string, int> getMajorityLabel(Table table) {
			string majorLabel = "";
			int majorCount = 0;

			map<string, int> labelCount;
			for(int i=0;i< table.data.size(); i++) {
				if(table.data[i].size() > 0) {
					labelCount[table.data[i].back()]++;
					if(labelCount[table.data[i].back()] > majorCount) {
						majorCount = labelCount[table.data[i].back()];
						majorLabel = table.data[i].back();
					}
				}
			}

			return {majorLabel, majorCount};
		}

		bool isLeafNode(Table table) {
			if(table.data.size() == 0) return true;
			
			for(int i=1;i < table.data.size();i++) {
				if(table.data[i].size() == 0 || table.data[0].size() == 0) {
					continue;
				}
				if(table.data[0].back() != table.data[i].back()) {
					return false;
				}
			}
			return true;
		}

		int getSelectedAttribute(Table table) {
			int maxAttrIndex = -1;
			double maxAttrValue = 0.0;

			if(initialTable.attrName.size() <= 1) {
				return -1;
			}

			// Usar CUDA apenas para tabelas grandes
			if(table.data.size() > 5000) {
				return getSelectedAttributeCUDA(table);
			}

			// Fallback para CPU
			for(int i=0; i< initialTable.attrName.size()-1; i++) {
				double gainRatio = getGainRatio(table, i);
				if(maxAttrValue < gainRatio) {
					maxAttrValue = gainRatio;
					maxAttrIndex = i;
				}
			}

			return maxAttrIndex;
		}

		int getSelectedAttributeCUDA(Table table) {
			int maxAttrIndex = -1;
			double maxAttrValue = 0.0;

			int numAttributes = initialTable.attrName.size() - 1;
			int numSamples = table.data.size();
			
			// Mapear classes
			map<string, int> classMap;
			vector<string> classNames;
			for(int i=0; i<table.data.size(); i++) {
				string className = table.data[i].back();
				if(classMap.find(className) == classMap.end()) {
					classMap[className] = classNames.size();
					classNames.push_back(className);
				}
			}
			int numClasses = classNames.size();

			// Calcular ganho de informação para cada atributo
			for(int attrIndex=0; attrIndex<numAttributes; attrIndex++) {
				double gainRatio = calculateGainRatioCUDA(table, attrIndex, classMap, numClasses);
				
				if(gainRatio > maxAttrValue) {
					maxAttrValue = gainRatio;
					maxAttrIndex = attrIndex;
				}
			}

			return maxAttrIndex;
		}

		double calculateGainRatioCUDA(Table& table, int attrIndex, 
		                              map<string, int>& classMap, int numClasses) {
			int numSamples = table.data.size();
			
			// Mapear valores do atributo
			map<string, int> attrValueMap;
			vector<string> attrValues;
			for(int i=0; i<numSamples; i++) {
				if(attrIndex < table.data[i].size()) {
					string attrValue = table.data[i][attrIndex];
					if(attrValueMap.find(attrValue) == attrValueMap.end()) {
						attrValueMap[attrValue] = attrValues.size();
						attrValues.push_back(attrValue);
					}
				}
			}
			int numAttrValues = attrValues.size();
			
			if(numAttrValues == 0) return 0.0;

			// Preparar dados para GPU
			vector<int> h_attrData(numSamples);
			vector<int> h_classData(numSamples);
			
			for(int i=0; i<numSamples; i++) {
				if(attrIndex < table.data[i].size()) {
					h_attrData[i] = attrValueMap[table.data[i][attrIndex]];
				} else {
					h_attrData[i] = -1;
				}
				h_classData[i] = classMap[table.data[i].back()];
			}

			// Copiar para device
			thrust::device_vector<int> d_attrData(h_attrData);
			thrust::device_vector<int> d_classData(h_classData);
			thrust::device_vector<int> d_attrValueCounts(numAttrValues, 0);
			thrust::device_vector<int> d_attrValueClassCounts(numAttrValues * numClasses, 0);

			// Lançar kernel
			int blockSize = BLOCK_SIZE;
			int gridSize = (numSamples + blockSize - 1) / blockSize;
			
			calculateAttrStats<<<gridSize, blockSize>>>(
				thrust::raw_pointer_cast(d_attrData.data()),
				thrust::raw_pointer_cast(d_classData.data()),
				numSamples, numClasses, numAttrValues,
				thrust::raw_pointer_cast(d_attrValueCounts.data()),
				thrust::raw_pointer_cast(d_attrValueClassCounts.data())
			);
			
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());

			// Copiar resultados de volta
			thrust::host_vector<int> h_attrValueCounts = d_attrValueCounts;
			thrust::host_vector<int> h_attrValueClassCounts = d_attrValueClassCounts;

			// Calcular ganho de informação na CPU
			double infoD = 0.0;
			map<int, int> classCounts;
			for(int i=0; i<numSamples; i++) {
				classCounts[h_classData[i]]++;
			}
			for(auto& p : classCounts) {
				double prob = (double)p.second / numSamples;
				if(prob > 0) {
					infoD -= prob * log2(prob);
				}
			}

			double infoAttrD = 0.0;
			double splitInfo = 0.0;
			
			for(int av=0; av<numAttrValues; av++) {
				int avCount = h_attrValueCounts[av];
				if(avCount == 0) continue;
				
				double avProb = (double)avCount / numSamples;
				
				// Split info
				if(avProb > 0) {
					splitInfo -= avProb * log2(avProb);
				}
				
				// Info for this attribute value
				double avInfo = 0.0;
				for(int c=0; c<numClasses; c++) {
					int count = h_attrValueClassCounts[av * numClasses + c];
					if(count > 0) {
						double prob = (double)count / avCount;
						avInfo -= prob * log2(prob);
					}
				}
				
				infoAttrD += avProb * avInfo;
			}

			double gain = infoD - infoAttrD;
			double gainRatio = (splitInfo > 0) ? (gain / splitInfo) : 0.0;
			
			return gainRatio;
		}

		double getGainRatio(Table table, int attrIndex) {
			double splitInfo = getSplitInfoAttrD(table, attrIndex);
			if(splitInfo == 0) return 0;
			return getGain(table, attrIndex)/splitInfo;
		}

		double getInfoD(Table table) {
			double ret = 0.0;
			int itemCount = (int)table.data.size();
			if(itemCount == 0) return 0;
			
			map<string, int> labelCount;
			for(int i=0;i<table.data.size();i++) {
				if(table.data[i].size() > 0) {
					labelCount[table.data[i].back()]++;
				}
			}

			for(auto iter=labelCount.begin(); iter != labelCount.end(); iter++) {
				double p = (double)iter->second/itemCount;
				if(p > 0) {
					ret += -1.0 * p * log(p)/log(2);
				}
			}
			return ret;
		}

		double getInfoAttrD(Table table, int attrIndex) {
			double ret = 0.0;
			int itemCount = (int)table.data.size();
			if(itemCount == 0) return 0;

			map<string, vector<int> > attrValueMap;
			for(int i=0;i<table.data.size();i++) {
				if(attrIndex < table.data[i].size()) {
					attrValueMap[table.data[i][attrIndex]].push_back(i);
				}
			}

			for(auto iter=attrValueMap.begin(); iter != attrValueMap.end(); iter++) {
				Table nextTable;
				for(int i=0;i<iter->second.size(); i++) {
					if(iter->second[i] < table.data.size()) {
						nextTable.data.push_back(table.data[iter->second[i]]);
					}
				}
				int nextItemCount = (int)nextTable.data.size();
				if(nextItemCount > 0) {
					ret += (double)nextItemCount/itemCount * getInfoD(nextTable);
				}
			}
			return ret;
		}

		double getGain(Table table, int attrIndex) {
			return getInfoD(table)-getInfoAttrD(table, attrIndex);
		}

		double getSplitInfoAttrD(Table table, int attrIndex) {
			double ret = 0.0;
			int itemCount = (int)table.data.size();
			if(itemCount == 0) return 0;

			map<string, vector<int> > attrValueMap;
			for(int i=0;i<table.data.size();i++) {
				if(attrIndex < table.data[i].size()) {
					attrValueMap[table.data[i][attrIndex]].push_back(i);
				}
			}

			for(auto iter=attrValueMap.begin(); iter != attrValueMap.end(); iter++) {
				Table nextTable;
				for(int i=0;i<iter->second.size(); i++) {
					if(iter->second[i] < table.data.size()) {
						nextTable.data.push_back(table.data[iter->second[i]]);
					}
				}
				int nextItemCount = (int)nextTable.data.size();
				if(nextItemCount > 0) {
					double d = (double)nextItemCount/itemCount;
					if(d > 0) {
						ret += -1.0 * d * log(d) / log(2);
					}
				}
			}
			return ret;
		}

		void printTree(int nodeIndex, string branch) {
			if(nodeIndex >= tree.size()) return;
			if (tree[nodeIndex].isLeaf == true) return;

			for(int i = 0; i < tree[nodeIndex].children.size(); i++) {
				int childIndex = tree[nodeIndex].children[i];
				if(childIndex >= tree.size()) continue;
				
				if(tree[nodeIndex].criteriaAttrIndex < initialTable.attrName.size()) {
					string attributeName = initialTable.attrName[tree[nodeIndex].criteriaAttrIndex];
					string attributeValue = tree[childIndex].attrValue;
					printTree(childIndex, branch + attributeName + " = " + attributeValue + ", ");
				}
			}
		}
};

class InputReader {
	private:
		ifstream fin;
		Table table;
	public:
		InputReader(string filename) {
			fin.open(filename);
			if(!fin) {
				cout << "Erro: arquivo " << filename << " não encontrado!" << endl;
				exit(0);
			}
			parse();
		}
		
		vector<string> parseCSVLine(string line) {
			vector<string> row;
			string field = "";
			bool inQuotes = false;
			
			for(int i = 0; i < line.size(); i++) {
				char c = line[i];
				if(c == '"') {
					inQuotes = !inQuotes;
				} else if(c == ',' && !inQuotes) {
					row.push_back(field);
					field = "";
				} else {
					field += c;
				}
			}
			row.push_back(field);
			return row;
		}
		
		void parse() {
			string str;
			bool isAttrName = true;
			
			while(getline(fin, str)){
				if(str.empty()) continue;
				if(!str.empty() && str.back() == '\r') {
					str.pop_back();
				}
				
				vector<string> row = parseCSVLine(str);
				if(isAttrName) {
					table.attrName = row;
					isAttrName = false;
				} else {
					if(row.size() == table.attrName.size()) {
						table.data.push_back(row);
					}
				}
			}
		}
		
		Table getTable() {
			return table;
		}
};

class DataSplitter {
	public:
		static pair<Table, Table> split(Table& fullTable, double trainRatio = 0.8) {
			Table trainTable, testTable;
			trainTable.attrName = fullTable.attrName;
			testTable.attrName = fullTable.attrName;
			
			vector<int> indices;
			for(int i = 0; i < fullTable.data.size(); i++) {
				indices.push_back(i);
			}
			
			unsigned seed = chrono::system_clock::now().time_since_epoch().count();
			shuffle(indices.begin(), indices.end(), default_random_engine(seed));
			
			int trainSize = (int)(fullTable.data.size() * trainRatio);
			
			for(int i = 0; i < trainSize; i++) {
				trainTable.data.push_back(fullTable.data[indices[i]]);
			}
			
			for(int i = trainSize; i < indices.size(); i++) {
				testTable.data.push_back(fullTable.data[indices[i]]);
			}
			
			return {trainTable, testTable};
		}
};

class OutputPrinter {
	private:
		ofstream fout;
	public:
		OutputPrinter(string filename) {
			fout.open(filename);
		}

		string joinByComma(vector<string> row) {
			string ret = "";
			for(int i=0; i< row.size(); i++) {
				ret += row[i];
				if(i != row.size() -1) ret += ',';
			}
			return ret;
		}

		void addLine(string str) {
			fout << str << endl;
		}
};

class AccuracyCalculator {
	public:
		static void calculate(DecisionTree& dt, Table& testTable) {
			int totalCount = testTable.data.size();
			int hitCount = 0;
			
			map<string, int> classCount;
			map<string, int> correctCount;
			
			cout << "TESTANDO O MODELO:" << endl << endl;
			
			for(int i = 0; i < testTable.data.size(); i++) {
				vector<string> testRow = testTable.data[i];
				string actualLabel = testRow.back();
				testRow.pop_back();
				string predictedLabel = dt.guess(testRow);
				
				classCount[actualLabel]++;
				if(actualLabel == predictedLabel) {
					hitCount++;
					correctCount[actualLabel]++;
				}
			}
			
			double accuracy = (double)hitCount / totalCount * 100;
			
			cout << "RESULTADOS:" << endl;
			cout << "  Acertos: " << hitCount << " / " << totalCount << endl;
			cout << "  Acurácia Geral: " << fixed << accuracy << "%" << endl << endl;
			
			cout << "ACURÁCIA POR CLASSE:" << endl;
			for(auto iter = classCount.begin(); iter != classCount.end(); iter++) {
				string className = iter->first;
				int total = iter->second;
				int correct = correctCount[className];
				double classAcc = (double)correct / total * 100;
				cout << "  " << className << ": " << classAcc << "%" << endl;
			}
		}
};

int main(int argc, const char * argv[]) {
	cout << "[INFO] Iniciando execução do programa..." << endl;
	
	if(argc < 3 || argc > 6) {
		cout << "Uso: ./dt <arquivo.csv> <coluna_alvo> [resultado.csv] [proporção_treino] [block_size]" << endl;
		cout << "  block_size: 32, 64, 128, 256, 512, 1024 (padrão: 256)" << endl;
		return 0;
	}

	string inputFileName = argv[1];
	string targetColumn = argv[2];
	string resultFileName = (argc >= 4) ? argv[3] : "resultado.csv";
	double trainRatio = (argc >= 5) ? atof(argv[4]) : 0.8;
	BLOCK_SIZE = (argc >= 6) ? atoi(argv[5]) : 256;
	
	// Validar BLOCK_SIZE
	if(BLOCK_SIZE < 32) BLOCK_SIZE = 32;
	if(BLOCK_SIZE > 1024) BLOCK_SIZE = 1024;
	
	if(trainRatio <= 0 || trainRatio >= 1) {
		cout << "Erro: proporção de treino deve estar entre 0 e 1" << endl;
		return 0;
	}

	cout << "[INFO] Lendo arquivo de entrada..." << endl;
	cout << "[CUDA] Block Size: " << BLOCK_SIZE << " threads/bloco" << endl;
	InputReader inputReader(inputFileName);
	Table fullTable = inputReader.getTable();
	cout << "[INFO] Tamanho da tabela lida: " << fullTable.data.size() << endl;
	
	if(fullTable.data.size() == 0) {
		cout << "Erro: arquivo vazio ou formato inválido!" << endl;
		return 0;
	}
	
	int targetIndex = -1;
	for(int i = 0; i < fullTable.attrName.size(); i++) {
		if(fullTable.attrName[i] == targetColumn) {
			targetIndex = i;
			break;
		}
	}
	
	if(targetIndex == -1) {
		cout << "Erro: coluna '" << targetColumn << "' não encontrada!" << endl;
		return 0;
	}
	
	if(targetIndex != fullTable.attrName.size() - 1) {
		string temp = fullTable.attrName[targetIndex];
		fullTable.attrName.erase(fullTable.attrName.begin() + targetIndex);
		fullTable.attrName.push_back(temp);
		
		for(int i = 0; i < fullTable.data.size(); i++) {
			string tempData = fullTable.data[i][targetIndex];
			fullTable.data[i].erase(fullTable.data[i].begin() + targetIndex);
			fullTable.data[i].push_back(tempData);
		}
	}
	
	pair<Table, Table> splitData = DataSplitter::split(fullTable, trainRatio);
	Table trainTable = splitData.first;
	Table testTable = splitData.second;

	cout << "[INFO] Treinando a árvore de decisão..." << endl;
	auto startTrain = chrono::high_resolution_clock::now();

	DecisionTree decisionTree(trainTable);

	auto endTrain = chrono::high_resolution_clock::now();
	auto durationTrain = chrono::duration_cast<chrono::milliseconds>(endTrain - startTrain);
	cout << "Tempo de treinamento: " << durationTrain.count() << " ms" << endl;

	cout << "[INFO] Testando o modelo..." << endl;
	AccuracyCalculator::calculate(decisionTree, testTable);

	cout << "\n[INFO] Gerando arquivo de resultado..." << endl;
	auto startOutput = chrono::high_resolution_clock::now();

	OutputPrinter outputPrinter(resultFileName);
	vector<string> outputHeader = testTable.attrName;
	outputHeader.push_back("predicted_" + outputHeader.back());
	outputPrinter.addLine(outputPrinter.joinByComma(outputHeader));

	for(int i = 0; i < testTable.data.size(); i++) {
		vector<string> result = testTable.data[i];
		vector<string> testRow = testTable.data[i];
		testRow.pop_back();
		result.push_back(decisionTree.guess(testRow));
		outputPrinter.addLine(outputPrinter.joinByComma(result));
	}

	auto endOutput = chrono::high_resolution_clock::now();
	auto durationOutput = chrono::duration_cast<chrono::milliseconds>(endOutput - startOutput);
	cout << "Tempo de geração de resultados: " << durationOutput.count() << " ms" << endl;

	auto totalDuration = chrono::duration_cast<chrono::milliseconds>(endOutput - startTrain);
	cout << "Tempo total de execução: " << totalDuration.count() << " ms" << endl;

	return 0;
}