package SPE;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Node class
 */
public class Node {

	/**
	 * node id
	 */
	private int id=-1;
	/**
	 * node type
	 */
	private String type=null;
	/**
	 * node type id
	 */
	private int typeId=-1;
	/**
	 * in neighbours
	 */
	public List<Node> in_nodes=new ArrayList<Node>();
	/**
	 * out neighbours
	 */
	public List<Node> out_nodes=new ArrayList<Node>();
	/**
	 * in neighbours ids
	 */
	public List<Integer> in_ids=new ArrayList<Integer>();
	/**
	 * out neighbours ids
	 */
	public List<Integer> out_ids=new ArrayList<Integer>();
	/**
	 * all neighbours
	 */
	public Set<Node> neighbours=new HashSet<Node>(); 

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}
	
	public int getTypeId() {
		return typeId;
	}

	public void setTypeId(int typeId) {
		this.typeId = typeId;
	}

	@Override
	public int hashCode() {
		// TODO Auto-generated method stub
		return this.id;
	}

	@Override
	public boolean equals(Object obj) {
		// TODO Auto-generated method stub
		if(obj instanceof Node){
			Node node=(Node) obj;
			if(node.getId()==this.id){
				return true;
			}
		}
		return false;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
//		return "id=="+id+",in_ids=="+in_ids.toString()+",out_ids=="+out_ids.toString();
		return "[id="+id+",neighbours=["+getNeighboursInfo()+"]]";
	}
	
	/**
	 * get nieghbours info
	 * @return
	 */
	private String getNeighboursInfo(){
		StringBuilder sb=new StringBuilder();
		if(neighbours.size()==0){
			return "";
		}
		else{
			for(Node n:neighbours){
				sb.append(n.id+",");
			}
			return sb.toString();
		}
	}
}
