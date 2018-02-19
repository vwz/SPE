package SPE;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;


/**
 * random walk sampling from source graph, and chenge them into user-only paths and then save them into a file.
 */
public class SamplingFromSourceGraph {

	/**
	 * random generator, to keep invariant
	 */
	private Random random=new Random(123);
	//nodes path
	static String nodesPath=Config.NODES_PATH;
	//edges path
	static String edgesPath=Config.EDGES_PATH;
	//type and typesid save file
	static String typeAndTypeIdPath=Config.TYPE_TYPEID_SAVEFILE;
	static String randomWalkSampling_savePath=Config.SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS;//file path to save random walk samplings
	static int K=Config.SAMPLING_TIMES_PER_NODE;//random walk sampling times for each node
	static int L=Config.SAMPLING_LENGTH_PER_PATH;//random walk sampling length for each path (or walker)
	static int shortest_path_length=Config.SHORTEST_LENGTH_FOR_SAMPLING;//the shortest length for paths in random walk sampling
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//get the whole graph
		ReadWholeGraph rwg=new ReadWholeGraph();
		Map<Integer,Node> graph=rwg.readDataFromFile(nodesPath, edgesPath, typeAndTypeIdPath);
		//sampling and then save
		SamplingFromSourceGraph sfa=new SamplingFromSourceGraph();
		sfa.randomWalkSampling(graph, K, L, randomWalkSampling_savePath);
	}

	/**
	 * random walk sampling
	 * @param data dataset
	 * @param k random walk sampling times for each node
	 * @param l random walk sampling length for each path (or walker)
	 * @param pathsFile savefile
	 */
	public void randomWalkSampling(Map<Integer,Node> data,int k,int l,String pathsSaveFile){
		List<Node> path=null;
		FileWriter writer=null;
		StringBuilder sb=new StringBuilder();
		try {
			writer=new FileWriter(pathsSaveFile);
		} catch (IOException e) {
			e.printStackTrace();
		}
		for(Node node:data.values()){
			for(int i=0;i<k;i++){
				path=randomWalkPath(node,l,-1,data);
				sb.delete( 0, sb.length() );
				int count=0;
				for(int j=0;j<path.size();j++){
					if(path.get(j).getType().equals("user")){
						sb.append(path.get(j).getId()+" ");
						count++;
					}
				}
				if(count<shortest_path_length){
					continue;
				}
				sb.append("\r\n");
				try {
					writer.write(sb.toString());
					writer.flush();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			
		}
	}
	
	/**
	 * get one path by random walk
	 * @param start start node
	 * @param l the length of this path
	 * @return
	 */
	private List<Node> randomWalkPath(Node start,int l,double prob_user,Map<Integer,Node> data){
		List<Node> path=new ArrayList<Node>(l+1);
		path.add(start);
		Node now=start;
		Set<Integer> types_set=new HashSet<Integer>();
		List<Integer> types=new ArrayList<Integer>();
		Map<Integer,List<Integer>> neighbours=new HashMap<Integer, List<Integer>>();
		int type=-1;
		List<Integer> list=null;
		for(int i=0;i<l;i++){
			if(now.out_nodes.size()==0){
				break;
			}
			types_set.clear();
			types.clear();
			neighbours.clear();
			for(Node n:now.out_nodes){
				types_set.add(n.getTypeId());
				if(neighbours.containsKey(n.getTypeId())){
					neighbours.get(n.getTypeId()).add(n.getId());
				}
				else{
					List<Integer> ids=new ArrayList<Integer>();
					ids.add(n.getId());
					neighbours.put(n.getTypeId(), ids);
				}
			}
			types.addAll(types_set);
			if(prob_user==-1){
				type=types.get(random.nextInt(types.size()));
				list=neighbours.get(type);
				now=data.get(list.get(random.nextInt(list.size())));
			}
			else{
				now=now.out_nodes.get(random.nextInt(now.out_nodes.size()));
			}
			path.add(now);
		}
		return path;
	}
}
