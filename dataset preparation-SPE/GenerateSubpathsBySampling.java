package SPE;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * generate user-only paths by ramdom walk samplings
 */
public class GenerateSubpathsBySampling {

	
	static String samplingsPath=Config.SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS;//the save file for random walk sampling
	static String subPathsSavePath=Config.SUBPATHS_SAVE_PATH;//the save file for subpaths(user-only paths) 
	static int window_maxlen=Config.LONGEST_ANALYSE_LENGTH_FOR_SAMPLING;//we use a window to analyse a path to generate subpath (o-path), this is the width of this window
	static int subpath_maxlen=Config.LONGEST_LENGTH_FOR_SUBPATHS;//the max length for a subpath (user-only path between two nodes)
	static int subpath_minlen=Config.SHORTEST_LENGTH_FOR_SUBPATHS;//the min length for a subpath (user-only path between two nodes)
	
	public static void main(String[] args) {
		
		GenerateSubpathsBySampling gsbs=new GenerateSubpathsBySampling();
		gsbs.generateSubPathsFromSamplings(samplingsPath, subPathsSavePath, window_maxlen, subpath_maxlen, subpath_minlen);
	}

	/**
	 * generate subpaths (user-only paths) between the query node q and another candidate node v.
	 */
	public void generateSubPathsFromSamplings(String samplingsPath, String subPathsSavePath,int window_maxlen,int subpath_maxlen,int subpath_minlen){
		BufferedReader br=null;
		String[] arr=null;
		FileWriter writer =null;
		String t=null;
		List<Integer> path=new ArrayList<Integer>();
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(samplingsPath), "UTF-8"));
			writer = new FileWriter(subPathsSavePath);
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();
				if(temp.length()>0){
					path.clear();
					arr=temp.split(" ");
					for(String s:arr){
						path.add(Integer.parseInt(s));
					}
					t=analyseOnePath(path, window_maxlen, subpath_maxlen, subpath_minlen);
					if(t.length()>0){
						writer.write(t);
						writer.flush();
					}
				}
			}
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		finally{
			try {
				if(writer!=null){
					writer.close();
					writer=null;
				}
				if(br!=null){
					br.close();
					br=null;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * generate all the subpaths for a given random walk sampling path
	 */
	private String analyseOnePath(List<Integer> path, int maxWindowLen, int maxSubpathLen, int subpath_minlen){
		StringBuilder sb=new StringBuilder();
		List<Integer> subpath=new ArrayList<Integer>();
		for(int i=0;i<path.size();i++){
			for(int j=i+1;j<path.size();j++){
				if(maxWindowLen>0 && (j-i)>maxWindowLen){
					break;
				}
				
				subpath.clear();
				for(int x=i;x<=j;x++){
					subpath.add(path.get(x)+0);
				}
				List<Integer> subpathNoRepeat=deleteRepeat(subpath);
				if(subpathNoRepeat.size()<subpath_minlen){
					subpathNoRepeat=null;
					continue;
				}
				
				if(maxSubpathLen>0 && subpathNoRepeat.size()>maxSubpathLen){
					continue;
				}
				
				sb.append(path.get(i)+"\t"+path.get(j)+"\t");
				for(int x=0;x<subpathNoRepeat.size();x++){
					sb.append(subpathNoRepeat.get(x)+" ");
				}
				sb.append("\r\n");
				subpathNoRepeat=null;
			}
		}
		return sb.toString();
	}
	
	/**
	 * delete repeat segments in one path
	 */
	public List<Integer> deleteRepeat(List<Integer> path){
		Map<Integer,Integer> map=new HashMap<Integer,Integer>();
		int node=0;
		List<Integer> result=new ArrayList<Integer>();
		int formerIndex=0;
		for(int i=0;i<path.size();i++){
			node=path.get(i);
			if(!map.containsKey(node)){
				map.put(node, i);
			}
			else{
				formerIndex=map.get(node);
				for(int j=formerIndex;j<i;j++){
					map.remove(path.get(j));
					path.set(j, -1);
				}
				map.put(node, i);
			}
		}
		for(int i=0;i<path.size();i++){
			if(path.get(i)!=-1){
				result.add(path.get(i));
			}
		}
		return result;
	}
}
