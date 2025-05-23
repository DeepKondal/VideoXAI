from neo4j import GraphDatabase

class ProvenanceModel:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_dataset(self, dataset_id, name, path):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Dataset {id: $dataset_id})
                SET d.name = $name, d.path = $path
                """,
                dataset_id=dataset_id, name=name, path=path
            )

    def create_pipeline_run(self, run_id, timestamp, status):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (p:PipelineRun {id: $run_id})
                SET p.timestamp = $timestamp, p.status = $status
                """,
                run_id=run_id, timestamp=timestamp, status=status
            )

    def create_processing_step(self, step_name, step_type, config):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (s:ProcessingStep {name: $step_name})
                SET s.step_type = $step_type, s.config = $config
                """,
                step_name=step_name, step_type=step_type, config=config
            )

    def link_pipeline_step(self, run_id, step_name):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:PipelineRun {id: $run_id}), (s:ProcessingStep {name: $step_name})
                MERGE (p)-[:CONTAINS]->(s)
                """,
                run_id=run_id, step_name=step_name
            )

    def link_dataset_to_processing(self, dataset_id, step_name):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (d:Dataset {id: $dataset_id}), (s:ProcessingStep {name: $step_name})
                MERGE (d)-[:PROCESSED_BY]->(s)
                """,
                dataset_id=dataset_id, step_name=step_name
            )


    def create_model_prediction(self, video_file, prediction):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (m:ModelPrediction {video_file: $video_file})
                SET m.prediction = $prediction, m.name = "Prediction"
                """,
                video_file=video_file, prediction=prediction
            )

    def link_processing_to_prediction(self, step_name, video_file):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (s:ProcessingStep {name: $step_name}), (m:ModelPrediction {video_file: $video_file})
                MERGE (s)-[:PRODUCED]->(m)
                """,
                step_name=step_name, video_file=video_file
            )

    def link_pipeline_to_prediction(self, run_id, video_file):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:PipelineRun {id: $run_id}), (m:ModelPrediction {video_file: $video_file})
                MERGE (p)-[:CONTAINS]->(m)
                """,
                run_id=run_id, video_file=video_file
            )


    def update_pipeline_status(self, run_id, new_status):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:PipelineRun {id: $run_id})
                SET p.status = $new_status
                """,
                run_id=run_id, new_status=new_status
            )

    def link_dataset_to_prediction(self, dataset_id, video_file):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (d:Dataset {id: $dataset_id}), (m:ModelPrediction {video_file: $video_file})
                MERGE (d)-[:GENERATED]->(m)
                """,
                dataset_id=dataset_id, video_file=video_file
            )
    '''
    def create_adversarial_prediction(self, video_file, prediction):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (m:AdversarialPrediction {video_file: $video_file})
                SET m.prediction = $prediction, m.name = "Adversarial Prediction"
                """,
                video_file=video_file, prediction=prediction
            )
    '''

    # 🔹 Create Prediction Node for Adversarial Videos
    def create_adversarial_prediction(self, video_file, prediction):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (m:AdversarialPrediction {video_file: $video_file})
                SET m.prediction = $prediction, m.type = "Adversarial"
                """,
                video_file=video_file, prediction=prediction
            )

    # 🔹 Link Processing Step to Predictions (Handles Both Original & Adversarial)
    def link_processing_to_prediction(self, step_name, video_file, is_adversarial=False):
        with self.driver.session() as session:
            node_type = "AdversarialPrediction" if is_adversarial else "ModelPrediction"
            session.run(
                f"""
                MATCH (s:ProcessingStep {{name: $step_name}}), (m:{node_type} {{video_file: $video_file}})
                MERGE (s)-[:PRODUCED]->(m)
                """,
                step_name=step_name, video_file=video_file
            )

    # 🔹 Link Pipeline to Predictions
    def link_pipeline_to_prediction(self, run_id, video_file, is_adversarial=False):
        with self.driver.session() as session:
            node_type = "AdversarialPrediction" if is_adversarial else "ModelPrediction"
            session.run(
                f"""
                MATCH (p:PipelineRun {{id: $run_id}}), (m:{node_type} {{video_file: $video_file}})
                MERGE (p)-[:CONTAINS]->(m)
                """,
                run_id=run_id, video_file=video_file
            )

    # 🔹 Link Dataset to Predictions
    def link_dataset_to_prediction(self, dataset_id, video_file, is_adversarial=False):
        with self.driver.session() as session:
            node_type = "AdversarialPrediction" if is_adversarial else "ModelPrediction"
            session.run(
                f"""
                MATCH (d:Dataset {{id: $dataset_id}}), (m:{node_type} {{video_file: $video_file}})
                MERGE (d)-[:GENERATED]->(m)
                """,
                dataset_id=dataset_id, video_file=video_file
            )

    # 🔹 Update Pipeline Status
    def update_pipeline_status(self, run_id, new_status):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:PipelineRun {id: $run_id})
                SET p.status = $new_status
                """,
                run_id=run_id, new_status=new_status
            )