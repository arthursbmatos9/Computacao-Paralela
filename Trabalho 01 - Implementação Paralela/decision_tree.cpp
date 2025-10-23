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
using namespace std;

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
			
			// Verificar se o índice é válido
			if(criteriaAttrIndex >= row.size()) {
				return -1;
			}

			for(int i=0;i<tree[here].children.size(); i++) {
				int next = tree[here].children[i];
				
				// Verificar se o próximo nó é válido
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
			
			// Verificar se encontrou um atributo válido
			if(selectedAttrIndex == -1) {
				tree[nodeIndex].isLeaf = true;
				tree[nodeIndex].label = getMajorityLabel(table).first;
				return;
			}

			map<string, vector<int> > attrValueMap;
			for(int i=0;i<table.data.size();i++) {
				// Verificar se o índice é válido
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

			// Verificar se o índice do atributo selecionado é válido
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
					// Verificar se o índice é válido
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
					// Copiar nomes dos atributos para a próxima tabela
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

			// Verificar se há atributos para selecionar
			if(initialTable.attrName.size() <= 1) {
				return -1;
			}

			for(int i=0; i< initialTable.attrName.size()-1; i++) {
				double gainRatio = getGainRatio(table, i);
				if(maxAttrValue < gainRatio) {
					maxAttrValue = gainRatio;
					maxAttrIndex = i;
				}
			}

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

// As outras classes (InputReader, DataSplitter, OutputPrinter, AccuracyCalculator) 
// permanecem as mesmas...

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
				
				// Remove \r se existir (Windows)
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
				
			}
		}
};

int main(int argc, const char * argv[]) {
	if(argc < 3 || argc > 5) {
		cout << "Uso: " << endl;
		cout << "  ./dt <arquivo.csv> <coluna_alvo> [resultado.csv] [proporção_treino]" << endl;
		cout << endl;
		cout << "Exemplos:" << endl;
		cout << "  ./dt cars.csv owner" << endl;
		cout << "  ./dt cars.csv owner resultado.csv" << endl;
		cout << "  ./dt cars.csv owner resultado.csv 0.7" << endl;
		cout << endl;
		cout << "A coluna_alvo é a classe que você quer prever (ex: owner, fuel, transmission)" << endl;
		return 0;
	}

	string inputFileName = argv[1];
	string targetColumn = argv[2];
	string resultFileName = (argc >= 4) ? argv[3] : "resultado.csv";
	double trainRatio = (argc >= 5) ? atof(argv[4]) : 0.8;
	
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
	
	// Encontrar índice da coluna alvo
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
	
	// Mover coluna alvo para o final
	if(targetIndex != fullTable.attrName.size() - 1) {
		// Trocar no header
		string temp = fullTable.attrName[targetIndex];
		fullTable.attrName.erase(fullTable.attrName.begin() + targetIndex);
		fullTable.attrName.push_back(temp);
		
		// Trocar em cada linha de dados
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
	
	// Iniciar timer do treinamento
	auto startTrain = chrono::high_resolution_clock::now();
	
	DecisionTree decisionTree(trainTable);
	
	// Finalizar timer do treinamento
	auto endTrain = chrono::high_resolution_clock::now();
	auto durationTrain = chrono::duration_cast<chrono::milliseconds>(endTrain - startTrain);
	
	cout << "Tempo de treinamento: " << durationTrain.count() << " ms" << endl;

	AccuracyCalculator::calculate(decisionTree, testTable);

	cout << "\nGERANDO ARQUIVO DE RESULTADO..." << endl;
	
	// Iniciar timer da geração de resultados
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
	
	// Finalizar timer da geração de resultados
	auto endOutput = chrono::high_resolution_clock::now();
	auto durationOutput = chrono::duration_cast<chrono::milliseconds>(endOutput - startOutput);
	
	cout << "Tempo de geração de resultados: " << durationOutput.count() << " ms" << endl;
	
	// Timer total
	auto totalDuration = chrono::duration_cast<chrono::milliseconds>(endOutput - startTrain);
	cout << "Tempo total de execução: " << totalDuration.count() << " ms" << endl;

	return 0;
}