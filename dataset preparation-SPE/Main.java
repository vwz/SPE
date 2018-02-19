package SPE;

import java.util.Date;
import java.util.Map;
import java.util.Set;

/**
 * Prepare datasets offline for MPE
 */
public class Main {

	static String nodesPath = Config.NODES_PATH;
	static String edgesPath = Config.EDGES_PATH;
	static String typeAndTypeIdPath = Config.TYPE_TYPEID_SAVEFILE;

	static String root_dir = Config.ROOT;// dataset root dir
	static String dataset_name = Config.DATASET_NAME;// dataset set name, e.g.
														// linkedin, facebook
	static String relation_class = Config.RELATION_CLASS;// relation class name,
															// e.g. classmate,
															// family
	static String allQueryPairs = Config.ALL_QUERY_PAIRS_PATH;// query pairs
																// save path

	static String subpathsFile = Config.SUBPATHS_SAVE_PATH;
	static String newSubpathsSaveFile = Config.NEW_SUBPATHS_SAVE_PATH;
	static String index2UserpairSaveFile = Config.INDEX_USERPAIR_SAVE_PATH;
	static String vectorSaveFile = Config.VECTOR_SAVE_PATH;

	static String instanceFolder = Config.INSTANCE_FOLDER;
	static String statNumSaveFile = Config.INSTANCE_STAT_NUM_SAVE_PATH;

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		long starttime = System.currentTimeMillis();

		// sample paths from source graph
		SamplingFromSourceGraph.main(null);

		// generate subpaths from sampling paths
		GenerateSubpathsBySampling.main(null);

		// For convenience, we get all the (q,v) tuples from training and test
		// data. So we only need to store less data.
		AnalyseTrainingAndTestData atatd = new AnalyseTrainingAndTestData();
		Set<String> queryTuples = atatd.analyseQueryTuples(root_dir, dataset_name, relation_class, allQueryPairs);

		// generate m-paths, and calculate the instances number between q and v
		GenerateSubgraphAndFeatureVector gmfv = new GenerateSubgraphAndFeatureVector();

		// find all the user-pairs
		gmfv.analyseSubpathsAndChnForFbAndLiUndirected(queryTuples, subpathsFile, newSubpathsSaveFile,
				index2UserpairSaveFile);
		// calculate the subgraph instances number and save
		gmfv.analyseInstancesGenerateStatNum(instanceFolder, statNumSaveFile);
		// generate the final m-node
		gmfv.generateVectorForUndirected(vectorSaveFile);

		System.out.println("Finished。。。Time == " + new Date());
		long endtime = System.currentTimeMillis();
		System.out.println("Cost time == " + (endtime - starttime) / 1000 + " s");
	}
}
