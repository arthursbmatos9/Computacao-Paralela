/*
 * VERSÃO HÍBRIDA (OpenMP + MPI) - ÁRVORE DE DECISÃO
 * 
 * MODIFICAÇÕES PARA PARALELIZAÇÃO:
 * 1. Adicionado suporte a MPI para distribuição de dados entre processos
 * 2. Paralelização OpenMP no cálculo de ganho de informação (getSelectedAttribute)
 * 3. Paralelização OpenMP na fase de teste/predição do modelo
 * 4. Paralelização OpenMP no parsing de linhas CSV
 * 5. Balanceamento de carga entre processos MPI
 */

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
#include <mpi.h>      // ADICIONADO: Suporte MPI
#include <omp.h>      // ADICIONADO: Suporte OpenMP
using namespace std;

class Table {
	public:
		vector<string> attrName;
		vector<vector<string> > data;

		vector<vector<string> > attrValueList;
		void extractAttrValue() {
			attrValueList.resize(attrName.size());
			
			// PARALELIZAÇÃO OPENMP: Paralelização do loop de extração de valores de atributos
			#pragma omp parallel for
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

			// PARALELIZAÇÃO OPENMP: Cálculo paralelo do ganho de informação para cada atributo
			vector<double> gainRatios(initialTable.attrName.size()-1);
			
			#pragma omp parallel for
			for(int i=0; i< initialTable.attrName.size()-1; i++) {
				gainRatios[i] = getGainRatio(table, i);
			}
			
			// Encontrar o melhor atributo sequencialmente (operação rápida)
			for(int i=0; i< initialTable.attrName.size()-1; i++) {
				if(maxAttrValue < gainRatios[i]) {
					maxAttrValue = gainRatios[i];
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
			vector<string> lines; // MODIFICAÇÃO MPI: Armazenar todas as linhas
			
			// Ler todas as linhas primeiro
			while(getline(fin, str)){
				lineNumber++;
				if(str.empty()) continue;
				
				// Remove \r se existir (Windows)
				if(!str.empty() && str.back() == '\r') {
					str.pop_back();
				}
				
				if(isAttrName) {
					vector<string> row = parseCSVLine(str);
					table.attrName = row;
					isAttrName = false;
				} else {
					lines.push_back(str);
				}
			}
			
			// PARALELIZAÇÃO OPENMP: Processing de linhas CSV em paralelo
			vector<vector<string>> parsedRows(lines.size());
			
			#pragma omp parallel for
			for(int i = 0; i < lines.size(); i++) {
				parsedRows[i] = parseCSVLine(lines[i]);
			}
			
			// Adicionar linhas válidas à tabela
			for(int i = 0; i < parsedRows.size(); i++) {
				if(parsedRows[i].size() == table.attrName.size()) {
					table.data.push_back(parsedRows[i]);
				}
			}
		}
		
		Table getTable() {
			return table;
		}
};

class DataSplitter {
	public:
		// MODIFICAÇÃO MPI: Dividir dados entre processos MPI
		static pair<Table, Table> splitMPI(Table& fullTable, int rank, int size, double trainRatio = 0.8) {
			Table trainTable, testTable;
			trainTable.attrName = fullTable.attrName;
			testTable.attrName = fullTable.attrName;
			
			vector<int> indices;
			for(int i = 0; i < fullTable.data.size(); i++) {
				indices.push_back(i);
			}
			
			// Apenas o processo 0 faz o shuffle
			if(rank == 0) {
				unsigned seed = chrono::system_clock::now().time_since_epoch().count();
				shuffle(indices.begin(), indices.end(), default_random_engine(seed));
			}
			
			// COMUNICAÇÃO MPI: Broadcast dos índices embaralhados
			MPI_Bcast(indices.data(), indices.size(), MPI_INT, 0, MPI_COMM_WORLD);
			
			int trainSize = (int)(fullTable.data.size() * trainRatio);
			
			// Dividir dados de treino entre processos
			int chunkSize = trainSize / size;
			int start = rank * chunkSize;
			int end = (rank == size - 1) ? trainSize : start + chunkSize;
			
			for(int i = start; i < end; i++) {
				if(i < indices.size() && indices[i] < fullTable.data.size()) {
					trainTable.data.push_back(fullTable.data[indices[i]]);
				}
			}
			
			// Dados de teste ficam com todos os processos (para validação)
			for(int i = trainSize; i < indices.size(); i++) {
				if(indices[i] < fullTable.data.size()) {
					testTable.data.push_back(fullTable.data[indices[i]]);
				}
			}
			
			return {trainTable, testTable};
		}
		
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
			
			// PARALELIZAÇÃO OPENMP: Predições em paralelo com redução
			#pragma omp parallel for reduction(+:hitCount)
			for(int i = 0; i < testTable.data.size(); i++) {
				vector<string> testRow = testTable.data[i];
				string actualLabel = testRow.back();
				
				testRow.pop_back();
				string predictedLabel = dt.guess(testRow);
				
				// Área crítica para atualizar contadores compartilhados
				#pragma omp critical
				{
					classCount[actualLabel]++;
					if(actualLabel == predictedLabel) {
						correctCount[actualLabel]++;
					}
				}
				
				if(actualLabel == predictedLabel) {
					hitCount++; // Esta variável usa reduction, então é thread-safe
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
				
				cout << "  " << className << ": " << correct << "/" << total 
					 << " (" << fixed << classAcc << "%)" << endl;
			}
		}
};

int main(int argc, char* argv[]) {
	// INICIALIZAÇÃO MPI
	MPI_Init(&argc, &argv);
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if(argc < 3 || argc > 5) {
		if(rank == 0) {
			cout << "Uso: " << endl;
			cout << "  mpirun -np <num_processos> ./dt_hybrid <arquivo.csv> <coluna_alvo> [resultado.csv] [proporção_treino]" << endl;
			cout << endl;
			cout << "Exemplos:" << endl;
			cout << "  mpirun -np 4 ./dt_hybrid CANCELLED_DIVERTED_2023.csv Cancelled" << endl;
			cout << "  mpirun -np 2 ./dt_hybrid CANCELLED_DIVERTED_2023.csv Cancelled resultado.csv" << endl;
			cout << "  mpirun -np 4 ./dt_hybrid CANCELLED_DIVERTED_2023.csv Cancelled resultado.csv 0.7" << endl;
			cout << endl;
			cout << "A coluna_alvo é a classe que você quer prever (ex: Cancelled, Diverted)" << endl;
		}
		MPI_Finalize();
		return 0;
	}

	string inputFileName = argv[1];
	string targetColumn = argv[2];
	string resultFileName = (argc >= 4) ? argv[3] : "resultado_hybrid.csv";
	double trainRatio = (argc >= 5) ? atof(argv[4]) : 0.8;
	
	if(trainRatio <= 0 || trainRatio >= 1) {
		if(rank == 0) {
			cout << "Erro: proporção de treino deve estar entre 0 e 1" << endl;
		}
		MPI_Finalize();
		return 0;
	}

	// Apenas o processo 0 lê o arquivo completo
	Table fullTable;
	if(rank == 0) {
		InputReader inputReader(inputFileName);
		fullTable = inputReader.getTable();
		
		if(fullTable.data.size() == 0) {
			cout << "Erro: arquivo vazio ou formato inválido!" << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
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
			MPI_Abort(MPI_COMM_WORLD, 1);
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
		
		cout << "EXECUTANDO VERSÃO HÍBRIDA (OpenMP + MPI)" << endl;
		cout << "Processos MPI: " << size << endl;
		cout << "Threads OpenMP por processo: " << omp_get_max_threads() << endl;
		cout << "Dataset: " << fullTable.data.size() << " registros, " 
			 << fullTable.attrName.size() << " atributos" << endl << endl;
	}
	
	// COMUNICAÇÃO MPI: Broadcast do tamanho dos dados
	int dataSize = 0, attrSize = 0;
	if(rank == 0) {
		dataSize = fullTable.data.size();
		attrSize = fullTable.attrName.size();
	}
	MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&attrSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	// Inicializar estruturas em todos os processos
	if(rank != 0) {
		fullTable.attrName.resize(attrSize);
		fullTable.data.resize(dataSize);
		for(int i = 0; i < dataSize; i++) {
			fullTable.data[i].resize(attrSize);
		}
	}
	
	// COMUNICAÇÃO MPI: Broadcast dos nomes dos atributos
	for(int i = 0; i < attrSize; i++) {
		int len = 0;
		if(rank == 0) len = fullTable.attrName[i].length();
		MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		if(rank != 0) fullTable.attrName[i].resize(len);
		MPI_Bcast((char*)fullTable.attrName[i].c_str(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
	}
	
	// COMUNICAÇÃO MPI: Broadcast dos dados
	for(int i = 0; i < dataSize; i++) {
		for(int j = 0; j < attrSize; j++) {
			int len = 0;
			if(rank == 0) len = fullTable.data[i][j].length();
			MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
			
			if(rank != 0) fullTable.data[i][j].resize(len);
			MPI_Bcast((char*)fullTable.data[i][j].c_str(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
		}
	}
	
	// Dividir dados entre processos MPI
	pair<Table, Table> splitData = DataSplitter::splitMPI(fullTable, rank, size, trainRatio);
	Table trainTable = splitData.first;
	Table testTable = splitData.second;

	if(rank == 0) {
		cout << "TREINANDO A ÁRVORE DE DECISÃO..." << endl;
	}
	
	// Iniciar timer do treinamento
	auto startTrain = chrono::high_resolution_clock::now();
	
	// Cada processo treina com seu subconjunto de dados
	DecisionTree decisionTree(trainTable);
	
	// Finalizar timer do treinamento
	auto endTrain = chrono::high_resolution_clock::now();
	auto durationTrain = chrono::duration_cast<chrono::milliseconds>(endTrain - startTrain);
	
	if(rank == 0) {
		cout << "Tempo de treinamento (Processo 0): " << durationTrain.count() << " ms" << endl;
	}

	// Todos os processos testam o modelo (cada um com sua árvore)
	AccuracyCalculator::calculate(decisionTree, testTable);

	// Apenas o processo 0 gera o arquivo de resultado
	if(rank == 0) {
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
	}
	
	// FINALIZAÇÃO MPI
	MPI_Finalize();
	return 0;
}