package SPE;

import java.io.FileInputStream;
import java.util.Properties;

/**
 * Config parameters for MPE
 */
public class Config {

	/**
	 * dataset root dir
	 */
	public static String ROOT="D:/dataset/icde2016/dataset/";
	/**
	 * dataset name
	 * facebook
	 * linkedin
	 */
	public static String DATASET_NAME="linkedin";
	/**
	 * relation name
	 * classmate
	 * family
	 * schoolmate
	 * colleague
	 */
	public static String RELATION_CLASS="colleague";
	/**
	 * Need to add graph node feature to m-node? default=False
	 */
	public static boolean IS_SUBGRAPH_COMBINE_FEATURE=false;
	/**
	 * Need to use m-node to replace the node feature? default=False
	 */
	public static boolean IS_SUBGRAPH_REPLACE_NODE_FEATURE=false;
	/**
	 * random walk sampling times for each node
	 */
	public static int SAMPLING_TIMES_PER_NODE=20;
	/**
	 * random walk sampling length for each path (or walker)
	 */
	public static int SAMPLING_LENGTH_PER_PATH=20;
	/**
	 * the shortest length for paths in random walk sampling
	 */
	public static int SHORTEST_LENGTH_FOR_SAMPLING=2;
	/**
	 * we use a window to analyse a path to generate subpath (o-path), this is the width of this window
	 */
	public static int LONGEST_ANALYSE_LENGTH_FOR_SAMPLING=20;
	/**
	 * the max length for a subpath (user-only path between two nodes)
	 */
	public static int LONGEST_LENGTH_FOR_SUBPATHS=10;
	/**
	 * the min length for a subpath (user-only path between two nodes)
	 */
	public static int SHORTEST_LENGTH_FOR_SUBPATHS=2;
	/**
	 * subgraph instance save path
	 */
	public static String INSTANCE_FOLDER="D:/SPE/subgraph-instances/instance-source-unzip-now/"+DATASET_NAME+"/subgraph.inst/";
	/**
	 * main work dir for a dataset, such as linkedin
	 */
	public static String MAIN_DIR=ROOT+DATASET_NAME+"/";
	/**
	 * the save path for subgraph instances number
	 */
	public static String INSTANCE_STAT_NUM_SAVE_PATH=MAIN_DIR+"statNumSaveFile";
	/**
	 * graph nodes file path
	 */
	public static String NODES_PATH=MAIN_DIR+"graph.node";
	/**
	 * graph edges file path
	 */
	public static String EDGES_PATH=MAIN_DIR+"graph.edge";
	/**
	 * type and typeid save file. ( this is a temp file )
	 */
	public static String TYPE_TYPEID_SAVEFILE=MAIN_DIR+"typeAndTypeIDSavePath";
	/**
	 * the save file for random walk sampling
	 */
	public static String SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS=MAIN_DIR+"randomWalkSamplingPaths";
	/**
	 * the save file for subpaths(user-only paths) .
	 */
	public static String SUBPATHS_SAVE_PATH=MAIN_DIR+"subpathsSaveFile";
	/**
	 * the save file for m-path.
	 */
	public static String NEW_SUBPATHS_SAVE_PATH=MAIN_DIR+"newSubpathsSaveFile";
	/**
	 * the IDs mapping relation for m-nodes
	 */
	public static String INDEX_USERPAIR_SAVE_PATH=MAIN_DIR+"index2UserpairSaveFile";
	/**
	 * the m-node embedding (vector) save file
	 */
	public static String VECTOR_SAVE_PATH=MAIN_DIR+"vectorSaveFile";
	/**
	 * the file to save all query pairs ( just for convenience)
	 */
	public static String ALL_QUERY_PAIRS_PATH="allQueryPairs-"+RELATION_CLASS;
	
	
}
