import os
import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PyPDF2 import PdfReader
import requests

@dataclass
class DocumentSection:
    """Represents a section of a legal document with metadata"""
    content: str
    section_type: str  # 'article', 'title', 'chapter', etc.
    section_number: Optional[str] = None
    title: Optional[str] = None
    page_number: Optional[int] = None
    word_count: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.content.split())

class EnhancedLegalChat:
    def __init__(self, llm_endpoint="http://localhost:11434/api/generate", model="llama3:latest"):
        self.documents = {}  # document_name -> List[DocumentSection]
        self.llm_endpoint = llm_endpoint
        self.model = model
        self.context_window_limit = 32000  # Adjust based on your LLM's context window
        
    def load_pdf_document(self, pdf_path: str, document_name: str = None) -> str:
        """Load and parse a PDF document with enhanced structure detection"""
        if document_name is None:
            document_name = os.path.basename(pdf_path)
            
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            
            # Extract text with page numbers
            pages_text = []
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                pages_text.append((page_num, text))
        
        # Parse document structure
        sections = self._parse_legal_structure(pages_text)
        self.documents[document_name] = sections
        
        print(f"Loaded document '{document_name}' with {len(sections)} sections")
        return document_name
    
    def _parse_legal_structure(self, pages_text: List[Tuple[int, str]]) -> List[DocumentSection]:
        """Enhanced parsing to maintain legal document structure"""
        sections = []
        current_section = ""
        current_page = 1
        section_type = "preamble"
        section_number = None
        section_title = None
        
        # Combine all text with page markers
        full_text = ""
        page_markers = {}
        for page_num, text in pages_text:
            page_markers[len(full_text)] = page_num
            full_text += f"\n[PAGE {page_num}]\n" + text
        
        # Enhanced regex patterns for Spanish legal documents
        patterns = {
            'title': r'(?:TÍTULO|Título)\s+([IVXLCDM]+|\d+)\.?\s*[:-]?\s*(.*?)(?=\n|$)',
            'chapter': r'(?:CAPÍTULO|Capítulo)\s+([IVXLCDM]+|\d+)\.?\s*[:-]?\s*(.*?)(?=\n|$)',
            'article': r'(?:Artículo|ARTÍCULO|Art\.?)\s+(\d+)\.?\s*[:-]?\s*(.*?)(?=\n|$)',
            'section': r'(?:Sección|SECCIÓN)\s+(\d+)\.?\s*[:-]?\s*(.*?)(?=\n|$)',
            'disposition': r'(?:Disposición|DISPOSICIÓN)\s+(.*?)(?=\n|$)'
        }
        
        lines = full_text.split('\n')
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('[PAGE'):
                if line.startswith('[PAGE'):
                    current_page = int(re.search(r'\[PAGE (\d+)\]', line).group(1))
                continue
                
            # Check if line matches any legal structure pattern
            matched = False
            for structure_type, pattern in patterns.items():
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous section if it exists
                    if current_content:
                        content = '\n'.join(current_content).strip()
                        if content:
                            sections.append(DocumentSection(
                                content=content,
                                section_type=section_type,
                                section_number=section_number,
                                title=section_title,
                                page_number=current_page
                            ))
                    
                    # Start new section
                    section_type = structure_type
                    section_number = match.group(1) if match.groups() else None
                    section_title = match.group(2).strip() if len(match.groups()) > 1 else None
                    current_content = [line]  # Include the header
                    matched = True
                    break
            
            if not matched:
                current_content.append(line)
        
        # Add final section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append(DocumentSection(
                    content=content,
                    section_type=section_type,
                    section_number=section_number,
                    title=section_title,
                    page_number=current_page
                ))
        
        return sections
    
    def search_documents(self, query: str, document_name: str = None) -> List[DocumentSection]:
        """Search documents using simple text matching and relevance scoring"""
        query_words = set(query.lower().split())
        results = []
        
        documents_to_search = [document_name] if document_name else self.documents.keys()
        
        for doc_name in documents_to_search:
            if doc_name not in self.documents:
                continue
                
            for section in self.documents[doc_name]:
                content_words = set(section.content.lower().split())
                
                # Simple relevance scoring
                common_words = query_words.intersection(content_words)
                if common_words:
                    relevance = len(common_words) / len(query_words)
                    # Boost score for exact phrase matches
                    if query.lower() in section.content.lower():
                        relevance *= 2
                    
                    section.relevance_score = relevance
                    results.append(section)
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:10]  # Return top 10 most relevant sections
    
    def _select_relevant_context(self, sections: List[DocumentSection], query: str, max_tokens: int = 8000) -> str:
        """Intelligently select context that fits within token limits"""
        context_parts = []
        current_tokens = 0
        
        # Estimate tokens (rough approximation: 1 token ≈ 0.75 words)
        def estimate_tokens(text: str) -> int:
            return int(len(text.split()) * 0.75)
        
        for i, section in enumerate(sections):
            section_text = f"\n[Sección {i+1}: {section.section_type.upper()}"
            if section.section_number:
                section_text += f" {section.section_number}"
            if section.title:
                section_text += f" - {section.title}"
            section_text += f"]\n{section.content}\n"
            
            section_tokens = estimate_tokens(section_text)
            
            if current_tokens + section_tokens > max_tokens:
                if not context_parts:  # If first section is too long, truncate it
                    words = section.content.split()
                    truncated = ' '.join(words[:max_tokens])
                    section_text = f"\n[Sección 1: {section.section_type.upper()}]\n{truncated}...\n"
                    context_parts.append(section_text)
                break
            
            context_parts.append(section_text)
            current_tokens += section_tokens
        
        return ''.join(context_parts)
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using local LLM"""
        try:
            resp = requests.post(self.llm_endpoint,
                               json={
                                   "model": self.model,
                                   "prompt": prompt,
                                   "stream": False
                               },
                               timeout=120)
            return resp.json().get("response", "Error: No response from LLM")
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def answer_question(self, question: str, document_name: str = None) -> Dict[str, any]:
        """Answer a question with enhanced context and cross-referencing"""
        # Search for relevant sections
        relevant_sections = self.search_documents(question, document_name)
        
        if not relevant_sections:
            return {
                "answer": "No he encontrado información relevante en los documentos cargados para responder a tu pregunta.",
                "sources": [],
                "context_used": ""
            }
        
        # Select context that fits within limits
        context = self._select_relevant_context(relevant_sections, question)
        
        # Create enhanced prompt with better instructions
        prompt = f"""Eres un asistente jurídico especializado en derecho español. Tienes acceso al contenido completo de documentos legales.

INSTRUCCIONES:
1. Responde ÚNICAMENTE basándote en el contexto proporcionado
2. Cita específicamente las secciones relevantes usando [Sección X]
3. Si encuentras información relacionada en diferentes secciones, menciónalo
4. Si no hay información suficiente, indícalo claramente
5. Proporciona una respuesta estructurada y precisa

PREGUNTA: {question}

CONTEXTO LEGAL:
{context}

RESPUESTA:"""

        answer = self._generate_response(prompt)
        
        # Prepare source information
        sources = []
        for i, section in enumerate(relevant_sections[:5]):  # Top 5 sources
            source_info = {
                "section_number": i + 1,
                "type": section.section_type,
                "number": section.section_number,
                "title": section.title,
                "relevance": getattr(section, 'relevance_score', 0),
                "content_preview": section.content[:200] + "..." if len(section.content) > 200 else section.content
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": context,
            "sections_found": len(relevant_sections)
        }
    
    def get_document_summary(self, document_name: str) -> Dict[str, any]:
        """Get a structured summary of a document"""
        if document_name not in self.documents:
            return {"error": f"Document '{document_name}' not found"}
        
        sections = self.documents[document_name]
        
        # Group by section type
        by_type = {}
        for section in sections:
            if section.section_type not in by_type:
                by_type[section.section_type] = []
            by_type[section.section_type].append(section)
        
        summary = {
            "document_name": document_name,
            "total_sections": len(sections),
            "section_types": {k: len(v) for k, v in by_type.items()},
            "total_words": sum(s.word_count for s in sections),
            "structure": []
        }
        
        # Create structure overview
        for section_type, type_sections in by_type.items():
            type_info = {
                "type": section_type,
                "count": len(type_sections),
                "items": []
            }
            
            for section in type_sections[:10]:  # Show first 10 of each type
                item = {
                    "number": section.section_number,
                    "title": section.title,
                    "word_count": section.word_count
                }
                type_info["items"].append(item)
            
            summary["structure"].append(type_info)
        
        return summary

# Usage example and testing
if __name__ == "__main__":
    # Initialize the enhanced chat system
    chat = EnhancedLegalChat()
    
    # Load a legal document
    try:
        doc_name = chat.load_pdf_document('docs/ley_arrendamiento_urbanos.pdf', 'Ley Arrendamientos Urbanos')
        
        # Get document summary
        summary = chat.get_document_summary(doc_name)
        print("RESUMEN DEL DOCUMENTO:")
        print(f"- Total de secciones: {summary['total_sections']}")
        print(f"- Palabras totales: {summary['total_words']}")
        print(f"- Tipos de sección: {summary['section_types']}")
        
        # Ask questions
        questions = [
            "¿Por cuánto tiempo puedo renovar mi contrato de alquiler?",
            "¿Cuáles son las obligaciones del arrendador?",
            "¿Qué pasa si no pago el alquiler?"
        ]
        
        for question in questions:
            print(f"\n{'='*50}")
            print(f"PREGUNTA: {question}")
            print('='*50)
            
            result = chat.answer_question(question, doc_name)
            print(f"RESPUESTA:\n{result['answer']}")
            
            if result['sources']:
                print(f"\nFUENTES CONSULTADAS ({result['sections_found']} secciones encontradas):")
                for source in result['sources'][:3]:  # Show top 3 sources
                    print(f"- {source['type'].upper()} {source['number'] or ''}: {source['title'] or 'Sin título'}")
    
    except FileNotFoundError:
        print("Error: No se encontró el archivo PDF. Asegúrate de que esté en la carpeta 'docs/'")
    except Exception as e:
        print(f"Error: {e}")
