#ifndef TSP_H
#define TSP_H
#include <bits/stdc++.h>
#include <vector>
#include <map>
#include <set>
#include <utility> 
#include <time.h> 
#include <stdlib.h> 

class Graph
{
private:
	int V; 
	int total_edges; 
	int initial_vertex; 
	std::map<std::pair<int, int>, float> map_edges; 
public:
	Graph(int V, int initial_vertex, bool random_graph = false); 
	void addEdge(int v1, int v2, float weight); 
	void showGraph(); 
	void generatesGraph(); 
	void showInfoGraph(); 
	float edgeCost(int src, int dest); 
	friend class Genetic; 
};
typedef std::pair<std::vector<int>, float> my_pair;

struct sort_pred
{
	bool operator()(const my_pair& firstElem, const my_pair& secondElem)
	{
		return firstElem.second < secondElem.second;
	}
};

class Genetic
{
private:
	Graph* graph; 
	std::vector< my_pair > population; 
	int size_population; 
	int real_size_population; 
	int generations; 
	int mutation_rate; 
	bool show_population; 
private:
	void initialPopulation(); 
public:
	Genetic(Graph* graph, int amount_population, int generations, int mutation_rate, bool show_population = true); 
	float isValidSolution(std::vector<int>& solution); 
	void showPopulation(); 
	void crossOver(std::vector<int>& parent1, std::vector<int>& parent2); 
	void insertBinarySearch(std::vector<int>& child, float total_cost); 
	void run(); 
	float getCostBestSolution(); 
	bool existsChromosome(const std::vector<int> & v); 
};

// credits :
// https: //github.com/parano/GeneticAlgorithm-TSP //

#endif
