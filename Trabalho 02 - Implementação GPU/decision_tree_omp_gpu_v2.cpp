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
#include <omp.h>
using namespace std;

// Variáveis globais para controlar modo de execução
bool USE_GPU_MODE = true;
int NUM_TEAMS = 32;  // Número de teams para GPU (apenas referência, pragma usa valor fixo)

// ============================================================================
// ESTRUTURAS PARA GPU
// ============================================================================

// Estrutura numérica da árvore para GPU
struct GPUTreeNode {
    int criteriaAttrIndex;
    int attrValueIndex;
    bool isLeaf;
    int labelIndex;
    int childrenStart;
    int numChildren;
};

// Estrutura para mapear strings para índices
struct StringMapper {
    map<string, int> stringToIndex;
    vector<string> indexToString;
    
    int getIndex(const string& str) {
        if(stringToIndex.find(str) == stringToIndex.end()) {
            int idx = indexToString.size();
            stringToIndex[str] = idx;
            indexToString.push_back(str);
        }
        return stringToIndex[str];
    }
    
    string getString(int idx) {
        if(idx >= 0 && idx < indexToString.size()) {
            return indexToString[idx];
        }
        return "Unknown";
    }
};

// ============================================================================
// FUNÇÕES AUXILIARES PARA GPU
// ============================================================================

#pragma omp declare target
double calc_entropy_gpu(int* counts, int total, int num_classes) {
    double entropy = 0.0;
    for(int i = 0; i < num_classes; i++) {
        if(counts[i] > 0) {
            double p = (double)counts[i] / total;
            entropy -= p * log(p) / log(2.0);
        }
    }
    return entropy;
}

// Função de predição na GPU (navegação iterativa)
int predictGPU(int* row, int numAttrs, GPUTreeNode* tree, int* childrenIndices, int nodeIdx) {
    int currentNode = nodeIdx;
    
    // Navegação iterativa (GPU não suporta recursão profunda)
    for(int depth = 0; depth < 100; depth++) {
        if(tree[currentNode].isLeaf) {
            return tree[currentNode].labelIndex;
        }
        
        int attrIdx = tree[currentNode].criteriaAttrIndex;
        if(attrIdx >= numAttrs || attrIdx < 0) {
            return -1;
        }
        
        int attrValue = row[attrIdx];
        int childStart = tree[currentNode].childrenStart;
        int numChildren = tree[currentNode].numChildren;
        
        // Procurar filho correspondente ao valor do atributo
        bool found = false;
        for(int i = 0; i < numChildren; i++) {
            int childIdx = childrenIndices[childStart + i];
            if(tree[childIdx].attrValueIndex == attrValue) {
                currentNode = childIdx;
                found = true;
                break;
            }
        }
        
        if(!found) {
            return -1;
        }
    }
    
    return -1; // Excedeu profundidade máxima
}
#pragma omp end declare target

// ============================================================================
// CLASSES PRINCIPAIS
// ============================================================================

class Table {
	public:
		vector<string> attrName;
		vector<vector<string> > data;

		vector<vector<string> > attrValueList;
		void extractAttrValue() {
			attrValueList.resize(attrName.size());

			// PARALELIZAÇÃO: Processamento host-side paralelo
            #pragma omp parallel for schedule(dynamic)
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

		// ========================================================================
		// NOVO MÉTODO: Converter árvore para formato GPU
		// ========================================================================
		void prepareGPUTree(vector<GPUTreeNode>& gpuTree, vector<int>& childrenIndices,
		                    vector<StringMapper>& attrMappers, StringMapper& labelMapper) {
		    
		    gpuTree.resize(tree.size());
		    int totalChildren = 0;
		    
		    // Primeira passagem: contar filhos totais
		    for(int i = 0; i < tree.size(); i++) {
		        totalChildren += tree[i].children.size();
		    }
		    childrenIndices.resize(totalChildren);
		    
		    // Preparar mappers para cada atributo
		    attrMappers.resize(initialTable.attrName.size());
		    for(int i = 0; i < initialTable.attrValueList.size(); i++) {
		        for(const auto& val : initialTable.attrValueList[i]) {
		            attrMappers[i].getIndex(val);
		        }
		    }
		    
		    // Segunda passagem: converter árvore
		    int childOffset = 0;
		    for(int i = 0; i < tree.size(); i++) {
		        gpuTree[i].criteriaAttrIndex = tree[i].criteriaAttrIndex;
		        gpuTree[i].isLeaf = tree[i].isLeaf;
		        
		        if(tree[i].isLeaf) {
		            gpuTree[i].labelIndex = labelMapper.getIndex(tree[i].label);
		            gpuTree[i].attrValueIndex = 0;
		            gpuTree[i].childrenStart = -1;
		            gpuTree[i].numChildren = 0;
		        } else {
		            // Converter valor do atributo para índice
		            if(tree[i].criteriaAttrIndex >= 0 && 
		               tree[i].criteriaAttrIndex < attrMappers.size()) {
		                if(!tree[i].attrValue.empty()) {
		                    gpuTree[i].attrValueIndex = 
		                        attrMappers[tree[i].criteriaAttrIndex].getIndex(tree[i].attrValue);
		                } else {
		                    gpuTree[i].attrValueIndex = 0;
		                }
		            }
		            
		            gpuTree[i].childrenStart = childOffset;
		            gpuTree[i].numChildren = tree[i].children.size();
		            
		            // Copiar índices dos filhos
		            for(int j = 0; j < tree[i].children.size(); j++) {
		                childrenIndices[childOffset + j] = tree[i].children[j];
		            }
		            childOffset += tree[i].children.size();
		        }
		    }
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

		double getEstimatedError(double f, int N) {
			double z = 0.69;
			if(N==0) {
				cout << ":: getEstimatedError :: N is zero" << endl;
				return 1.0;
			}
			return (f+z*z/(2*N)+z*sqrt(f/N-f*f/N+z*z/(4*N*N)))/(1+z*z/N);
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

			int numAttrs = initialTable.attrName.size() - 1;
			double* gainRatios = new double[numAttrs];
			
			// Calcular gain ratios em paralelo (CPU)
			#pragma omp parallel for schedule(dynamic)
			for(int i=0; i < numAttrs; i++) {
				gainRatios[i] = getGainRatio(table, i);
			}
			
			// OFFLOAD GPU - Redução para encontrar máximo
			if(USE_GPU_MODE) {
				#pragma omp target teams distribute parallel for num_teams(16) map(to: gainRatios[0:numAttrs]) reduction(max:maxAttrValue)
				for(int i=0; i < numAttrs; i++) {
					if(gainRatios[i] > maxAttrValue) {
						maxAttrValue = gainRatios[i];
					}
				}
			} else {
				#pragma omp parallel for reduction(max:maxAttrValue)
				for(int i=0; i < numAttrs; i++) {
					if(gainRatios[i] > maxAttrValue) {
						maxAttrValue = gainRatios[i];
					}
				}
			}
			
			// Encontrar índice do máximo (CPU)
			for(int i=0; i < numAttrs; i++) {
				if(fabs(gainRatios[i] - maxAttrValue) < 1e-9) {
					maxAttrIndex = i;
					break;
				}
			}
			
			delete[] gainRatios;

			return maxAttrIndex;
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
			
			if (tree[nodeIndex].isLeaf == true) {
				return;
			}

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
			int lineNumber = 0;
			
			while(getline(fin, str)){
				lineNumber++;
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
			if(!fout) {
				cout << "Erro ao criar arquivo " << filename << endl;
				exit(0);
			}
		}

		string joinByComma(vector<string> row) {
			string ret = "";
			for(int i=0; i< row.size(); i++) {
				ret += row[i];
				if(i != row.size() -1) {
					ret += ',';
				}
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
			
			vector<string> predictions(testTable.data.size());
			
			// Predição paralela na CPU
            #pragma omp parallel for schedule(dynamic)
			for(int i = 0; i < testTable.data.size(); i++) {
				vector<string> testRow = testTable.data[i];
				testRow.pop_back();
				predictions[i] = dt.guess(testRow);
			}
			
			// Cálculo de acurácia
			for(int i = 0; i < testTable.data.size(); i++) {
				string actualLabel = testTable.data[i].back();
				string predictedLabel = predictions[i];
				
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
				cout << "  " << className << ": " << fixed << classAcc << "% (" 
					 << correct << "/" << total << ")" << endl;
			}
		}
};

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, const char * argv[]) {
	if(argc < 3 || argc > 8) {
		cout << "Uso: " << endl;
		cout << "  ./dt <arquivo.csv> <coluna_alvo> [resultado.csv] [proporção_treino] [modo] [num_teams]" << endl;
		cout << endl;
		cout << "Parâmetros:" << endl;
		cout << "  arquivo.csv        - Dataset de entrada" << endl;
		cout << "  coluna_alvo        - Nome da coluna target" << endl;
		cout << "  resultado.csv      - Arquivo de saída (padrão: resultado.csv)" << endl;
		cout << "  proporção_treino   - Proporção treino/teste (padrão: 0.8)" << endl;
		cout << "  modo              - 'gpu' ou 'cpu' (padrão: gpu)" << endl;
		cout << "  num_teams         - Número de teams GPU: 32, 64, 128, 256 (padrão: 32)" << endl;
		cout << endl;
		cout << "Exemplos:" << endl;
		cout << "  ./dt cars.csv owner" << endl;
		cout << "  ./dt cars.csv owner resultado.csv" << endl;
		cout << "  ./dt cars.csv owner resultado.csv 0.7 cpu" << endl;
		cout << "  ./dt cars.csv owner resultado.csv 0.8 gpu 64" << endl;
		return 0;
	}

	string inputFileName = argv[1];
	string targetColumn = argv[2];
	string resultFileName = (argc >= 4) ? argv[3] : "resultado.csv";
	double trainRatio = (argc >= 5) ? atof(argv[4]) : 0.8;
	string mode = (argc >= 6) ? argv[5] : "gpu";
	NUM_TEAMS = (argc >= 7) ? atoi(argv[6]) : 32;
	
	transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
	USE_GPU_MODE = (mode == "gpu");
	
	// Validar NUM_TEAMS
	if(NUM_TEAMS < 1) NUM_TEAMS = 32;
	if(NUM_TEAMS > 1024) NUM_TEAMS = 1024;
	
	omp_set_num_threads(8);
	int num_devices = omp_get_num_devices();
	
	cout << "=========================================" << endl;
	cout << "   Decision Tree - Versão OpenMP" << endl;
	cout << "=========================================" << endl;
	cout << "Modo selecionado: " << (USE_GPU_MODE ? "GPU" : "CPU") << endl;
	cout << "Threads OpenMP: " << 8 << endl;
	cout << "Num Teams GPU: " << NUM_TEAMS << endl;
	cout << "Dispositivos GPU disponíveis: " << num_devices << endl;
	
	if(USE_GPU_MODE) {
		if(num_devices > 0) {
			omp_set_default_device(0);
			cout << "[GPU] Offload ativado para device 0" << endl;
			cout << "[GPU] Kernels numéricos executando na GPU" << endl;
		} else {
			cout << "[AVISO] GPU solicitada mas não detectada" << endl;
			cout << "[INFO] Offload será emulado no host" << endl;
		}
	} else {
		cout << "[CPU] Modo paralelo multi-core" << endl;
		cout << "[CPU] Sem offload GPU" << endl;
	}
	cout << "=========================================" << endl << endl;
	
	if(trainRatio <= 0 || trainRatio >= 1) {
		cout << "Erro: proporção de treino deve estar entre 0 e 1" << endl;
		return 0;
	}

	InputReader inputReader(inputFileName);
	Table fullTable = inputReader.getTable();
	
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

	cout << "TREINANDO A ÁRVORE DE DECISÃO..." << endl;
	
	auto startTrain = chrono::high_resolution_clock::now();
	
	DecisionTree decisionTree(trainTable);
	
	auto endTrain = chrono::high_resolution_clock::now();
	auto durationTrain = chrono::duration_cast<chrono::milliseconds>(endTrain - startTrain);
	
	cout << "Tempo de treinamento: " << durationTrain.count() << " ms" << endl;

	AccuracyCalculator::calculate(decisionTree, testTable);

	cout << "\nGERANDO ARQUIVO DE RESULTADO..." << endl;
	
	auto startOutput = chrono::high_resolution_clock::now();
	
	OutputPrinter outputPrinter(resultFileName);
	
	vector<string> outputHeader = testTable.attrName;
	outputHeader.push_back("predicted_" + outputHeader.back());
	outputPrinter.addLine(outputPrinter.joinByComma(outputHeader));
	
	// ========================================================================
	// PREDIÇÃO GPU - NOVA IMPLEMENTAÇÃO
	// ========================================================================
	
	vector<vector<string>> results(testTable.data.size());
	int n_results = testTable.data.size();
	
	if(USE_GPU_MODE && n_results > 0) {
		cout << "[GPU] Preparando dados para predição em batch..." << endl;
		
		// Preparar estruturas de conversão
		vector<StringMapper> attrMappers;
		StringMapper labelMapper;
		vector<GPUTreeNode> gpuTree;
		vector<int> childrenIndices;

		// Converter árvore para formato GPU
		decisionTree.prepareGPUTree(gpuTree, childrenIndices, attrMappers, labelMapper);
		
		// Converter dados de teste para numérico
		int numAttrs = testTable.attrName.size() - 1;  // Excluir label
		int* testDataFlat = new int[n_results * numAttrs];
		
		for(int i = 0; i < n_results; i++) {
			for(int j = 0; j < numAttrs && j < testTable.data[i].size() - 1; j++) {
				testDataFlat[i * numAttrs + j] = attrMappers[j].getIndex(testTable.data[i][j]);
			}
		}
		
		// Array de predições
		int* predictions = new int[n_results];
		for(int i = 0; i < n_results; i++) {
			predictions[i] = -1;
		}
		
		// Preparar ponteiros para GPU
		GPUTreeNode* d_tree = gpuTree.data();
		int* d_children = childrenIndices.data();
		int treeSize = gpuTree.size();
		int childrenSize = childrenIndices.size();
		
		cout << "[GPU] Executando " << n_results << " predições na GPU com 16 teams..." << endl;
		
		// PREDIÇÃO GPU - OFFLOAD REAL
		#pragma omp target teams distribute parallel for num_teams(16) \
		    map(to: testDataFlat[0:n_results*numAttrs], d_tree[0:treeSize], \
		            d_children[0:childrenSize], numAttrs) \
		    map(from: predictions[0:n_results])
		for(int i = 0; i < n_results; i++) {
			int* row = &testDataFlat[i * numAttrs];
			predictions[i] = predictGPU(row, numAttrs, d_tree, d_children, 0);
		}
		
		cout << "[GPU] Predições concluídas!" << endl;
		
		// Converter resultados de volta para strings
		#pragma omp parallel for
		for(int i = 0; i < n_results; i++) {
			results[i] = testTable.data[i];
			
			string prediction;
			if(predictions[i] >= 0) {
				prediction = labelMapper.getString(predictions[i]);
			} else {
				// Fallback: usar predição CPU se GPU falhou
				vector<string> testRow = testTable.data[i];
				testRow.pop_back();
				prediction = decisionTree.guess(testRow);
			}
			results[i].push_back(prediction);
		}
		
		delete[] testDataFlat;
		delete[] predictions;
		
	} else {
		// MODO CPU PARALELO (fallback)
		cout << "[CPU] Processando predições em paralelo..." << endl;
		
		#pragma omp parallel for schedule(dynamic)
		for(int i = 0; i < testTable.data.size(); i++) {
			results[i] = testTable.data[i];
			vector<string> testRow = testTable.data[i];
			testRow.pop_back();
			results[i].push_back(decisionTree.guess(testRow));
		}
	}

	// Escrita sequencial
	for(int i = 0; i < results.size(); i++) {
		outputPrinter.addLine(outputPrinter.joinByComma(results[i]));
	}

	auto endOutput = chrono::high_resolution_clock::now();
	auto durationOutput = chrono::duration_cast<chrono::milliseconds>(endOutput - startOutput);

	cout << "Tempo de geração de resultados: " << durationOutput.count() << " ms" << endl;

	auto totalDuration = chrono::duration_cast<chrono::milliseconds>(endOutput - startTrain);
	cout << "Tempo total de execução: " << totalDuration.count() << " ms" << endl;

	return 0;
}
