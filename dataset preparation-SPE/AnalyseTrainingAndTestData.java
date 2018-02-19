package SPE;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Set;

/**
 * For convenience, we get all the (q,v) from training data and test data, so that we only need to check and store the paths between q and v.
 */
public class AnalyseTrainingAndTestData {

	static String root_dir=Config.ROOT;//dataset root dir
	static String dataset_name=Config.DATASET_NAME;//dataset name, such as linkedin
	static String relation_class=Config.RELATION_CLASS;//relation class name, such as classmate
	static String allQueryPairs=Config.ALL_QUERY_PAIRS_PATH;//query pairs file
	
	static Set<String> allNodes=new HashSet<String>();
	
	public static void main(String[] args) {
		AnalyseTrainingAndTestData atatd=new AnalyseTrainingAndTestData();
		
		Set<String> set1=atatd.analyseQueryTuples(
				"D:/dataset/icde2016/dataset/", 
				"facebook", 
				"classmate", 
				"allQueryPairs");
		System.out.println(set1.size());
		Set<String> set2=atatd.analyseQueryTuples(
				"D:/dataset/icde2016/dataset/", 
				"facebook", 
				"family", 
				"allQueryPairs");
		System.out.println(set2.size());
		Set<String> set=new HashSet<String>();
		set.addAll(set1);
		set.addAll(set2);
		System.out.println(set.size());
		System.out.println(allNodes.size());
	}

	/**
	 * we get all the (q,a) from training and test dataset, and then save them.
	 */
	public Set<String> analyseQueryTuples(String mainFolder, String datasetName, String relationName, String saveFileName){
		String train_10=mainFolder+datasetName+".splits/train.10/";
		String train_100=mainFolder+datasetName+".splits/train.100/";
		String train_1000=mainFolder+datasetName+".splits/train.1000/";
		String test=mainFolder+datasetName+".splits/test/";
		Set<String> allPairs=new HashSet<String>();
		String filePath=null;
		for(int i=1;i<=10;i++){
			filePath=train_10+"train_"+relationName+"_"+i;
			Set<String> set=analyseQueryTuplesForOneFile(filePath);
			allPairs.addAll(set);
		}
		for(int i=1;i<=10;i++){
			filePath=train_100+"train_"+relationName+"_"+i;
			Set<String> set=analyseQueryTuplesForOneFile(filePath);
			allPairs.addAll(set);
		}
		for(int i=1;i<=10;i++){
			filePath=train_1000+"train_"+relationName+"_"+i;
			Set<String> set=analyseQueryTuplesForOneFile(filePath);
			allPairs.addAll(set);
		}
		for(int i=1;i<=10;i++){
			filePath=test+"test_"+relationName+"_"+i;
			Set<String> set=analyseQueryTuplesForOneFile(filePath);
			allPairs.addAll(set);
		}
		
		FileWriter writer =null;
		String saveFile=mainFolder+datasetName+"/"+saveFileName;
		try {
			writer = new FileWriter(saveFile);
			for(String pair:allPairs){
				writer.write(pair+"\r\n");
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
		return allPairs;
	}
	
	/**
	 * get all (q,a) from one file
	 */
	private Set<String> analyseQueryTuplesForOneFile(String filePath){
		Set<String> result=new HashSet<String>();
		BufferedReader br=null;
		String[] arr=null;
		try {//读文件
			br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), "UTF-8"));
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();
				if(temp.length()>0){
					arr=temp.split("\t");
					allNodes.add(arr[0]);
					for(int i=1;i<arr.length;i++){
						allNodes.add(arr[i]);
						result.add(arr[0]+"\t"+arr[i]);
					}
				}
			}
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		finally{
			try {
				if(br!=null){
					br.close();
					br=null;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return result;
	}
}
