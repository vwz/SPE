package SPE;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Find all user-pairs, and calculate the subgraph instances number and save, and generate the final m-nodes and m-paths.
 */
public class GenerateSubgraphAndFeatureVector {

	static String mainFolder="";
	static String datasetName="";
	static String relationName="";
	static String saveFileName="";
	
	static String subpathsFile="D:/test/modeling-subgraph/toydata-addUserEdges/subpathsSaveFile";
	static String newSubpathsSaveFile="D:/test/modeling-subgraph/toydata-addUserEdges/newSubpathsSaveFile";
	static String index2UserpairSaveFile="D:/test/modeling-subgraph/toydata-addUserEdges/index2UserpairSaveFile";
	static String vectorSaveFile="D:/test/modeling-subgraph/toydata-addUserEdges/vectorSaveFile";
	
	static Map<String,int[]> vectors=new HashMap<String, int[]>();
	static Map<String,Integer> userPair2Index=new HashMap<String,Integer>();
	static Map<Integer,String> index2UserPair=new HashMap<Integer,String>();
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
//		AnalyseTrainingAndTestData atatd=new AnalyseTrainingAndTestData();
//		Set<String> queryTuples=atatd.analyseQueryTuples(mainFolder, datasetName, relationName, saveFileName);
		
//		Set<String> queryTuples=new HashSet<String>();
//		queryTuples.add("1\t13");
//		queryTuples.add("5\t6");
//		
//		
		GenerateSubgraphAndFeatureVector gmfv=new GenerateSubgraphAndFeatureVector();
//		gmfv.analyseSubpathsAndChnForDblpDirected(queryTuples, subpathsFile, newSubpathsSaveFile, index2UserpairSaveFile);
//		System.out.println("OK");
//		System.out.println(vectors);
//		System.out.println(userPair2Index);
//		System.out.println(index2UserPair);
		
		index2UserPair.put(0, "0\t3");
		index2UserPair.put(1, "5\t2");
		index2UserPair.put(2, "4\t0");
		index2UserPair.put(3, "8\t9");
		index2UserPair.put(4, "0\t4");
		index2UserPair.put(5, "9\t8");
		userPair2Index.put("0\t3",0);
		userPair2Index.put("5\t2",1);
		userPair2Index.put("4\t0",2);
		userPair2Index.put("8\t9",3);
		userPair2Index.put("0\t4",4);
		userPair2Index.put("9\t8",5);
		vectors.put("0\t3", null);
		vectors.put("5\t2", null);
		vectors.put("4\t0", null);
		vectors.put("8\t9", null);
		vectors.put("0\t4", null);
		vectors.put("9\t8", null);
		vectors.put("0", null);
		vectors.put("2", null);
		vectors.put("3", null);
		vectors.put("4", null);
		vectors.put("5", null);
		vectors.put("8", null);
		vectors.put("9", null);
		gmfv.analyseInstancesGenerateStatNum("D:/test/modeling-subgraph/instancesToydata/","D:/test/modeling-subgraph/vectors");
		for(String key:vectors.keySet()){
			System.out.println(key);
			System.out.println(Arrays.toString(vectors.get(key)));
			System.out.println("-------------------------------");
		}
//		gmfv.generateVectorForUndirected(vectorSaveFile);
//		gmfv.generateVectorForDirectedWithNodeFeature("D:/test/modeling-subgraph/nodeFeature", vectorSaveFile);
		
//		gmfv.generateSingleNodeVectorBySubgraph("D:/test/modeling-subgraph/nodeFeature");
	}

	/**
	 * change user-only paths to m-paths
	 * @param queryTuples all query tuples (q,a)
	 * @param subpathsFile user-only subpahts
	 * @param newSubpathsSaveFile file for m-paths
	 * @param index2UserpairSaveFile  file for m-node IDs and the corresponding user-pairs
	 */
	public void analyseSubpathsAndChnForFbAndLiUndirected(Set<String> queryTuples, String subpathsFile, String newSubpathsSaveFile, String index2UserpairSaveFile){
		BufferedReader br=null;
		FileWriter writer = null;
		String[] arr=null;
		String[] arr1=null;
		String queryTuple1=null;
		String queryTuple2=null;
		String userpair1=null;
		String userpair2=null;
		StringBuilder sb=new StringBuilder();
		int index=0;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(subpathsFile), "UTF-8"));
			writer = new FileWriter(newSubpathsSaveFile);
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();
				if(temp.length()>0){
					arr=temp.split("\t");
					queryTuple1=arr[0]+"\t"+arr[1];
					queryTuple2=arr[1]+"\t"+arr[0];
					if(queryTuples.contains(queryTuple1) || queryTuples.contains(queryTuple2)){
						sb.delete( 0, sb.length() );
						sb.append(arr[0]+"\t"+arr[1]+"\t");
						arr1=arr[2].split(" ");
						for(int i=0;i<(arr1.length-1);i++){
							userpair1=arr1[i]+"\t"+arr1[i+1];
							userpair2=arr1[i+1]+"\t"+arr1[i];
							vectors.put(arr1[i], null);
							vectors.put(arr1[i+1], null);
							if(!vectors.containsKey(userpair1) && !vectors.containsKey(userpair2)){
								vectors.put(userpair1, null);
								userPair2Index.put(userpair1, userPair2Index.size());
								index2UserPair.put(index2UserPair.size(), userpair1);
								index=userPair2Index.get(userpair1);
							}
							else{
								if(vectors.containsKey(userpair1)){
									index=userPair2Index.get(userpair1);
								}
								else{
									index=userPair2Index.get(userpair2);
								}
							}
							sb.append(index+" ");
						}
						sb.append("\r\n");
						writer.write(sb.toString());
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
		
		try {
			writer = new FileWriter(index2UserpairSaveFile);
			for(int id:index2UserPair.keySet()){
				writer.write(id+"\t"+index2UserPair.get(id)+"\r\n");
				writer.flush();
			}
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		finally{
			try {
				if(writer!=null){
					writer.close();
					writer=null;
				}
			} catch (Exception e2) {
				// TODO: handle exception
				e2.printStackTrace();
			}
		}
	}
	
	
	/**
	 * analyse instances for subgraphs and get the statistical numbers
	 * @param instanceFolder instance folder
	 * @param statNumSaveFile statistical numbers file
	 */
	public void analyseInstancesGenerateStatNum(String instanceFolder,String statNumSaveFile){
		File folder = new File(instanceFolder);
		File[] files = folder.listFiles();
		int dimensionSubgraph=files.length;
		for(String key:vectors.keySet()){
			vectors.put(key, new int[dimensionSubgraph]);
		}
		String filePath=null;
		BufferedReader br=null;
		String[] arr=null;
		int[] intArr=null;
		String userPair1=null;
		String userPair2=null;
		for(int i=0;i<files.length;i++){
			filePath=instanceFolder+i;
			try {
				br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), "UTF-8"));
				String temp = null;
				while ((temp = br.readLine()) != null ) {
					temp=temp.trim();
					if(temp.length()>0){
						arr=temp.split("\t");
						for(String s:arr){
							if(vectors.containsKey(s)){
								intArr=vectors.get(s);
								intArr[i]+=1;
							}
						}
						for(int a=0;a<arr.length;a++){
							for(int b=a+1;b<arr.length;b++){
								userPair1=arr[a]+"\t"+arr[b];
								userPair2=arr[b]+"\t"+arr[a];
								if(vectors.containsKey(userPair1) || vectors.containsKey(userPair2)){
									if(vectors.containsKey(userPair1)){
										intArr=vectors.get(userPair1);
										intArr[i]+=1;
									}
									if(vectors.containsKey(userPair2)){
										intArr=vectors.get(userPair2);
										intArr[i]+=1;
									}
								}
							}
						}
					}
				}
			} catch (Exception e2) {
				// TODO: handle exception
				e2.printStackTrace();
			}
			finally{
				try {
					if(br!=null){
						br.close();
						br=null;
					}
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		FileWriter writer =null;
		StringBuilder sb=new StringBuilder();
		try {
			writer = new FileWriter(statNumSaveFile);
			for(String key:vectors.keySet()){
				intArr=vectors.get(key);
				sb.delete( 0, sb.length() );
				if(key.contains("\t")){
					sb.append(key+"\t");
				}
				else{
					sb.append(key+"\t-1\t");
				}
				for(int i=0;i<intArr.length;i++){
					sb.append(intArr[i]+"\t");
				}
				sb.append("\r\n");
				writer.write(sb.toString());
				writer.flush();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		finally{
			try {
				if(writer!=null){
					writer.close();
					writer=null;
				}
			} catch (Exception e2) {
				// TODO: handle exception
				e2.printStackTrace();
			}
		}
	}
	
	
	/**
	 * generate the embedding (or vector) for m-node
	 * @param vectorSaveFile save file
	 */
	public void generateVectorForUndirected(String vectorSaveFile){
		Map<Integer,double[]> result=new HashMap<Integer,double[]>();
		int index=0;
		String[] arr=null;
		double[] vector=null;
		int[] statNum=null;
		int[] statNum_0=null;
		int[] statNum_1=null;
		int dimension=0;
		for(String key:vectors.keySet()){
			if(key.contains("\t")){
				index=userPair2Index.get(key);
				statNum=vectors.get(key);
				dimension=statNum.length;
				vector=new double[statNum.length];
				arr=key.split("\t");
				statNum_0=vectors.get(arr[0]);
				statNum_1=vectors.get(arr[1]);
				for(int i=0;i<statNum.length;i++){
					vector[i]=computeWeight(statNum[i], statNum_0[i], statNum_1[i]);
				}
				result.put(index, vector);
			}
		}
		FileWriter writer =null;
		StringBuilder sb=new StringBuilder();
		try {
			writer = new FileWriter(vectorSaveFile);
			writer.write(result.size()+" "+dimension+"\r\n");
			writer.flush();
			for(int key:result.keySet()){
				sb.delete( 0, sb.length() );
				sb.append(key+" ");
				vector=result.get(key);
				for(int i=0;i<vector.length;i++){
					sb.append(vector[i]+" ");
				}
				sb.append("\r\n");
				writer.write(sb.toString());
				writer.flush();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		finally{
			try {
				if(writer!=null){
					writer.close();
					writer=null;
				}
			} catch (Exception e2) {
				// TODO: handle exception
				e2.printStackTrace();
			}
		}
	}
	
	/**
	 * compute weight
	 * @param ab
	 * @param a
	 * @param b
	 * @return
	 */
	private double computeWeight(int ab,int a,int b){
		if(ab>0){
			return 1.0;
		}
		return 0.0;
	}
}
